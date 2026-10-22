"""Durable shared API budget reservations at public, undiscounted token rates.

Reservations are planning upper estimates, not a guarantee of third-party
billing. With enforcement enabled, unexpected usage closes the gate; with
enforce=False, costs and overruns are recorded without an automatic spending stop.
Every attempt, including retries and interrupted attempts, keeps its own charge.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time


class BudgetExceeded(RuntimeError):
    """No new network attempt may be started under this ledger."""


def _number(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (integer and int(value) != value)):
        raise ValueError(f'{name} must be a finite nonnegative number' + (' of whole tokens.' if integer else '.'))
    return int(value) if integer else float(value)


def pricing(config):
    return (_number(config.get('price_input_per_million'), 'price_input_per_million'),
            _number(config.get('price_output_per_million'), 'price_output_per_million'))


def estimate_reservation(messages, config, *, image_count=None):
    """UTF-8 text bytes + explicit overhead + per-image allowance, no discounts."""
    texts, images = [], 0
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, str):
            texts.append(content)
        else:
            for item in content:
                if item.get('type') == 'text':
                    texts.append(item['text'])
                elif item.get('type') == 'image_url':
                    images += 1
    if image_count is not None:
        images = max(images, _number(image_count, 'image_count', True))
    overhead = _number(config.get('budget_input_overhead_tokens', 8192), 'budget_input_overhead_tokens', True)
    per_image = _number(config.get('budget_image_tokens', 4096), 'budget_image_tokens', True)
    estimated = sum(len(text.encode('utf-8')) for text in texts) + overhead + images * per_image
    estimated = max(estimated, _number(config.get('budget_max_input_tokens', 0), 'budget_max_input_tokens', True))
    requested_output = _number(config.get('max_tokens', 1024), 'max_tokens', True)
    if requested_output == 0:
        raise ValueError('max_tokens must be positive for a budgeted request.')
    output = max(requested_output, _number(config.get('budget_output_tokens_allowance', 0),
                                          'budget_output_tokens_allowance', True))
    in_rate, out_rate = pricing(config)
    return dict(input_tokens=estimated, output_tokens=output,
                input_per_million=in_rate, output_per_million=out_rate,
                reserved_usd=(estimated*in_rate + output*out_rate)/1e6,
                input_estimate_method='utf8_bytes_plus_overhead_and_original_image_allowance')


class BudgetTracker:
    """Cross-process file-locked ledger, reusable across models and event loops.

    Public methods are synchronous and short; async callers should use
    asyncio.to_thread. Request keys must identify model/config/input/condition.
    Completed response bodies are retained to recover a crash before JSONL write.
    """
    def __init__(self, path, limit_usd, *, enforce=True):
        self.path = Path(path).expanduser().resolve()
        if not isinstance(enforce, bool):
            raise ValueError('enforce must be Boolean.')
        self.enforce = enforce
        self.limit_usd = _number(limit_usd, 'limit_usd')
        if self.limit_usd <= 0:
            raise ValueError('limit_usd must be positive.')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            ledger = self._read()
            if not self.enforce and (ledger.get('enforced', True) or ledger['stopped']):
                ledger.setdefault('audit_events', []).append(dict(
                    event='disable_budget_enforcement', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    note='Spending protection disabled by explicit user authorization; preserve all historical charges.',
                    previous_stop_reason=ledger['stop_reason'], previous_stop_category=ledger.get('stop_category'),
                    committed_usd=self._total(ledger), historical_limit_usd=ledger['limit_usd']))
                ledger.update(stopped=False, stop_reason=None, stop_category=None)
            ledger['enforced'] = self.enforce
            self._write(ledger)

    @contextmanager
    def _locked(self):
        lock_path = self.path.with_name(self.path.name + '.lock')
        with lock_path.open('a+b') as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            if os.name == 'nt':
                import msvcrt
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(.01)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self):
        if not self.path.exists():
            return dict(schema_version=1, limit_usd=self.limit_usd, entries={}, stopped=False, stop_reason=None)
        ledger = json.loads(self.path.read_text('utf-8'))
        if ledger.get('schema_version') != 1 or (self.enforce and ledger.get('limit_usd') != self.limit_usd):
            raise ValueError('Budget ledger version/limit differs; use a new ledger for a new budget.')
        if not isinstance(ledger.get('entries'), dict):
            raise ValueError('Invalid budget ledger entries.')
        for entry in ledger['entries'].values():
            _number(entry['charged_usd'], 'ledger charged_usd')
        return ledger

    def _write(self, ledger):
        temporary = self.path.with_name(self.path.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(ledger, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    @staticmethod
    def _total(ledger):
        return math.fsum(entry['charged_usd'] for entry in ledger['entries'].values())

    def snapshot(self):
        with self._locked():
            ledger = self._read()
            total = self._total(ledger)
            active_stop = ledger['stopped'] and (self.enforce or ledger.get('stop_category') == 'operator_reconfigure')
            return dict(limit_usd=self.limit_usd if self.enforce else None, committed_usd=total,
                        historical_limit_usd=ledger['limit_usd'], enforced=self.enforce,
                        remaining_usd=max(0., self.limit_usd-total) if self.enforce else None, stopped=active_stop,
                        stop_reason=ledger['stop_reason'] if active_stop else None,
                        stop_category=ledger.get('stop_category') if active_stop else None,
                        attempts=len(ledger['entries']))

    def request_entries(self, request_key):
        with self._locked():
            return [dict(entry) for entry in self._read()['entries'].values()
                    if entry['request_key'] == request_key]

    def reserve(self, request_key, estimate, *, model=None):
        amount = _number(estimate['reserved_usd'], 'reserved_usd')
        with self._locked():
            ledger = self._read()
            if ledger['stopped'] and (self.enforce or ledger.get('stop_category') == 'operator_reconfigure'):
                # Preserve the first cause, including a reservation undercount;
                # another waiting worker must not relabel it as budget depletion.
                raise BudgetExceeded(ledger['stop_reason'] or 'The shared budget gate is stopped.')
            if self.enforce and self._total(ledger) + amount > self.limit_usd + 1e-12:
                ledger.update(stopped=True, stop_category='reservation_would_exceed_budget',
                              stop_reason='Next attempt reservation would exceed the shared budget.',
                              last_denied_reservation_usd=amount)
                self._write(ledger)
                raise BudgetExceeded(ledger['stop_reason'])
            prior = [e for e in ledger['entries'].values() if e['request_key'] == request_key]
            entry_id = f'{request_key}:{len(prior)+1}'
            ledger['entries'][entry_id] = dict(id=entry_id, request_key=request_key, model=model,
                status='reserved', reserved_usd=amount, charged_usd=amount, usage=None,
                **{key: estimate[key] for key in ('input_tokens','output_tokens','input_per_million','output_per_million')})
            self._write(ledger)
            return entry_id

    def settle(self, entry_id, *, response=None, failed=False):
        with self._locked():
            ledger = self._read()
            entry = ledger['entries'][entry_id]
            if entry['status'] != 'reserved':
                return dict(entry)
            usage = response.get('usage') if isinstance(response, dict) else None
            actual = None
            if not failed and isinstance(usage, dict):
                try:
                    prompt = _number(usage.get('prompt_tokens'), 'prompt_tokens', True)
                    completion = _number(usage.get('completion_tokens'), 'completion_tokens', True)
                    actual = (prompt*entry['input_per_million'] + completion*entry['output_per_million'])/1e6
                except ValueError:
                    pass
            entry.update(status='failed_retained' if failed else 'settled' if actual is not None else 'missing_usage_retained',
                         usage=usage, charged_usd=entry['reserved_usd'] if actual is None else actual)
            if not failed and response is not None:
                entry['response'] = response
            if actual is not None and actual > entry['reserved_usd'] + 1e-12:
                if self.enforce and (not ledger['stopped'] or ledger.get('stop_category') == 'reservation_overrun'):
                    ledger.update(stopped=True, stop_category='reservation_overrun',
                                  stop_reason='Observed usage exceeded its reservation; no further requests allowed.')
                entry['reservation_overrun_usd'] = actual-entry['reserved_usd']
            if self.enforce and self._total(ledger) > self.limit_usd + 1e-12:
                ledger.update(stopped=True, stop_category='actual_cost_exceeds_budget',
                              stop_reason='Observed committed cost exceeds the configured budget.')
            self._write(ledger)
            return dict(entry)

    def acknowledge_reservation_overrun(self, note, *, allow_legacy_reason_recovery=False):
        """Audit and reopen only an under-estimated reservation, never the cap.

        Call only after correcting token allowances and after all in-flight
        attempts settle. The opt-in legacy flag handles the early ledger schema
        whose first stop reason could be overwritten by another queued worker.
        Charges and the spending ceiling are never changed by this operation.
        """
        if not isinstance(note, str) or not note.strip():
            raise ValueError('A nonempty audit note describing corrected allowances is required.')
        with self._locked():
            ledger = self._read()
            total = self._total(ledger)
            if not ledger['stopped']:
                raise ValueError('Budget gate is not stopped.')
            if total >= self.limit_usd:
                raise ValueError('Cannot reopen a budget whose committed cost reached its limit.')
            if any(entry['status'] == 'reserved' for entry in ledger['entries'].values()):
                raise ValueError('Cannot reopen while an attempt remains unsettled.')
            acknowledged = set(ledger.get('acknowledged_overrun_ids', []))
            overruns = [key for key, entry in ledger['entries'].items()
                        if entry.get('reservation_overrun_usd', 0) > 0 and key not in acknowledged]
            if not overruns:
                raise ValueError('No unacknowledged reservation-overrun evidence exists.')
            category = ledger.get('stop_category')
            legacy = (allow_legacy_reason_recovery and category is None and
                      ledger['stop_reason'] in {
                          'Observed usage exceeded its reservation; no further requests allowed.',
                          'Next attempt reservation would exceed the shared budget.'})
            if category != 'reservation_overrun' and not legacy:
                raise ValueError('Only a reservation-overrun stop may be acknowledged; budget-limit stops cannot be reopened.')
            ledger.setdefault('audit_events', []).append(dict(
                event='acknowledge_reservation_overrun', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                note=note.strip(), previous_stop_reason=ledger['stop_reason'], previous_stop_category=category,
                legacy_reason_recovered=bool(legacy), overrun_entry_ids=overruns,
                committed_usd=total, unchanged_limit_usd=self.limit_usd))
            ledger['acknowledged_overrun_ids'] = sorted(acknowledged | set(overruns))
            ledger.update(stopped=False, stop_reason=None, stop_category=None)
            self._write(ledger)
            return dict(reopened=True, committed_usd=total, limit_usd=self.limit_usd,
                        acknowledged_overrun_ids=overruns, legacy_reason_recovered=bool(legacy))
