"""Offline checks of final-stage cohort freezing and incremental cost planning."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_final


class FinalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'results'/'final_run'
        self.out.mkdir(parents=True)
        self.config = dict(price_input_per_million=1_000_000, price_output_per_million=0)
        self.observed = [dict(image_paths=['image.png'], usage=dict(prompt_tokens=1,completion_tokens=0))]

    def test_projection_prices_only_pending_unique_request_ids(self):
        plan = dict(requests=[dict(id='first',image_paths=['a']),
                              dict(id='second',image_paths=['a','b']),
                              dict(id='third',image_paths=['a'])])
        cache = self.root/'cache.jsonl'
        cache.write_text('\n'.join(json.dumps(row) for row in [
            dict(question_id='first',condition='image',status='error'),
            dict(question_id='second',condition='text',status='ok'),
            dict(question_id='third',condition='image',status='invalid_response')])+'\n', 'utf-8')
        summary = run_final.ablation_cost_projection(plan, self.observed, self.config, cache)
        self.assertEqual(summary['completed_unique_requests'], 2)
        self.assertEqual(summary['pending_unique_requests'], 1)
        self.assertAlmostEqual(summary['estimated_usd_with_20pct_margin'], 2.4)
        # A partial last write is ignored for planning and repaired by evaluator.
        with cache.open('a', encoding='utf-8') as handle:
            handle.write('{"question_id":')
        self.assertEqual(summary, run_final.ablation_cost_projection(plan,self.observed,self.config,cache))

    def test_kimi_stays_at_30_without_cost_halt_and_completed_resume_costs_zero(self):
        run_final.save(self.out/'status.json', {'kimi':{'smoke_passed':True}})
        (self.out/'kimi_image.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.observed), 'utf-8')
        cache = self.root/'results'/'.ablation_cache'/'kimi.requests.jsonl'
        plans = []
        invoked = []
        class Tracker:
            remaining = 100
            def snapshot(self): return dict(remaining_usd=self.remaining, stopped=False)
        tracker = Tracker()
        def select(_questions, per_type, seed):
            return [dict(id=f'q-{i}') for i in range(per_type*3)]
        def build(chosen, scenes, seed):
            # 100 requests at 30/task exceed the old nominal remaining amount;
            # explicit authorization must prevent a downgrade or price halt.
            plan = dict(requests=[dict(id=f'request-{i}',image_paths=['a']) for i in range(len(chosen)+10)],
                        base_question_ids=[q['id'] for q in chosen])
            plans.append(plan)
            return plan
        async def execute(questions, scenes, config, path, **kwargs):
            invoked.append(kwargs)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(''.join(json.dumps(dict(question_id=q['id'],condition='image',status='ok'))+'\n'
                                     for q in plans[-1]['requests']), 'utf-8')
            return [None]*(kwargs['per_type']*3*16), dict(model='kimi',variant_records=kwargs['per_type']*3*16)
        with patch.object(run_final, 'ROOT', self.root), patch.object(run_final, 'OUT', self.out), \
             patch.object(run_final, 'load_questions', return_value=[]), \
             patch.object(run_final, 'load_scenes', return_value={}), \
             patch.object(run_final, 'model_config', return_value=self.config), \
             patch.object(run_final, 'select_questions', side_effect=select), \
             patch.object(run_final, 'build_ablation_plan', side_effect=build), \
             patch.object(run_final, 'run_ablation', side_effect=execute), \
             patch.object(run_final, 'emit'):
            asyncio.run(run_final.ablations(['kimi'], tracker))
            saved = json.loads((self.out/'ablation_kimi.plan.json').read_text())
            self.assertEqual(saved['per_type'],30)
            estimate = json.loads((self.out/'ablation_budget_estimates.json').read_text())['kimi']
            self.assertAlmostEqual(estimate['estimated_usd_with_20pct_margin'],120)
            self.assertEqual(invoked[0]['per_type'],30)
            self.assertEqual(invoked[0]['timeout'],300)
            self.assertEqual(invoked[0]['max_retries'],1)
            self.assertIs(invoked[0]['budget'],tracker)
            tracker.remaining = .001
            asyncio.run(run_final.ablations(['kimi'], tracker))
            resumed = json.loads((self.out/'ablation_budget_estimates.json').read_text())['kimi']
            self.assertEqual(resumed['estimated_usd_with_20pct_margin'],0)
            self.assertEqual(resumed['pending_unique_requests'],0)
            self.assertEqual(resumed['per_type'],30)
            self.assertEqual(len(invoked),2)
            self.assertEqual(saved,json.loads((self.out/'ablation_kimi.plan.json').read_text()))

    def test_main_conditions_are_explicit_and_smoke_reuses_same_full_manifest(self):
        from evaluate import _fingerprint
        questions = [dict(id=f'q-{i}',qtype='Q3') for i in range(200)]
        controls = [dict(id=f'control-{i}',qtype='VISION_CONTROL') for i in range(2)]
        calls = []
        class Tracker:
            def snapshot(self): return dict(remaining_usd=40,stopped=False)
        async def evaluate(qs, config, output, **kwargs):
            calls.append(dict(questions=qs,config=config,path=output,kwargs=kwargs))
            active = qs[:kwargs['limit']] if kwargs.get('limit') else qs
            return [dict(qtype=q['qtype'],status='ok',correct=True,parse_method='final_answer_line',
                         response_model='offline',latency_seconds=0,usage=None) for q in active]
        with patch.object(run_final,'ROOT',self.root), patch.object(run_final,'OUT',self.out), \
             patch.object(run_final,'ordered_mini',return_value=questions), \
             patch.object(run_final,'vision_controls',return_value=controls), \
             patch.object(run_final,'model_config',return_value=dict(alias='gpt',model='offline',provider='offline')), \
             patch.object(run_final,'evaluate_questions',side_effect=evaluate), patch.object(run_final,'emit'):
            asyncio.run(run_final.smoke(['gpt'],Tracker()))
            asyncio.run(run_final.main_table(['gpt'],Tracker()))
        smoke, main = calls[0],calls[2]
        self.assertEqual(smoke['path'],main['path'])
        self.assertEqual(_fingerprint(smoke['questions']),_fingerprint(main['questions']))
        self.assertEqual(len(smoke['questions']),200)
        self.assertEqual(smoke['kwargs']['limit'],8)
        self.assertEqual(smoke['kwargs']['seed'],main['kwargs']['seed'])
        model_calls = [call for call in calls[2:] if call['config']['alias'] == 'gpt']
        self.assertEqual([call['kwargs']['conditions'] for call in model_calls],
                         [('image',),('text',),('no_image',)])

    def test_parallel_pipeline_advances_independently_merges_status_and_resumes(self):
        models = ['gpt','claude','kimi']
        questions = [dict(id=f'q-{i}',qtype='Q3') for i in range(200)]
        controls = [dict(id=f'control-{i}',qtype='VISION_CONTROL') for i in range(2)]
        cache, ablation_done, events, baseline_calls = {},set(),[],[]
        active = peak = sent = 0
        claude_main_started = None
        class Tracker:
            def snapshot(self): return dict(remaining_usd=None,stopped=False,enforced=False)
        def config(alias):
            return dict(alias=alias,model=alias,provider='offline',
                        price_input_per_million=1_000_000,price_output_per_million=0)
        async def evaluate(qs, cfg, output, **kwargs):
            nonlocal active,peak,sent
            alias = cfg['alias']
            condition = kwargs.get('conditions',('image',))[0]
            phase = 'smoke' if kwargs.get('limit') else 'control' if qs[0]['qtype']=='VISION_CONTROL' else condition
            events.append((alias,phase,'start'))
            if alias != 'random':
                self.assertEqual(kwargs['concurrency'],2 if phase=='control' else 8)
                self.assertEqual(kwargs['timeout'],300)
                self.assertEqual(kwargs['max_retries'],1)
                self.assertEqual(kwargs['seed'],run_final.SEED)
                active += 1
                peak = max(peak,active)
                if alias == 'claude' and phase == 'image':
                    claude_main_started.set()
                if alias == 'kimi' and phase == 'smoke':
                    # Hold Kimi until Claude really enters main. A sequential
                    # pipeline would time out; machine speed cannot change the
                    # ordering assertion as it could with a fixed 25 ms delay.
                    await asyncio.wait_for(claude_main_started.wait(), timeout=5)
                else:
                    await asyncio.sleep(0)
                active -= 1
            else:
                baseline_calls.append(condition)
            selected = qs[:kwargs['limit']] if kwargs.get('limit') else qs
            output = Path(output)
            bucket = cache.setdefault(str(output),{})
            for q in selected:
                key = (q['id'],condition)
                if key not in bucket:
                    if alias != 'random': sent += 1
                    bucket[key] = dict(question_id=q['id'],condition=condition,qtype=q['qtype'],
                        status='ok',correct=True,parse_method='final_answer_line',response_model=alias,
                        latency_seconds=.001,usage=dict(prompt_tokens=1,completion_tokens=0),image_paths=['a'])
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(''.join(json.dumps(row)+'\n' for row in bucket.values()),'utf-8')
            events.append((alias,phase,'end'))
            return [bucket[(q['id'],condition)] for q in selected]
        def select(_questions,per_type,seed):
            self.assertEqual((per_type,seed),(30,20261021))
            return [dict(id=f'ab-{i}') for i in range(90)]
        def plan(chosen,scenes,seed):
            return dict(requests=[dict(id=q['id'],image_paths=['a']) for q in chosen],
                        base_question_ids=[q['id'] for q in chosen])
        async def ablate(qs,scenes,cfg,output,**kwargs):
            nonlocal sent,active,peak
            alias = cfg['alias']
            self.assertEqual((kwargs['per_type'],kwargs['concurrency'],kwargs['timeout'],kwargs['max_retries']),
                             (30,8,300,1))
            events.append((alias,'ablation','start'))
            active += 1
            peak = max(peak,active)
            await asyncio.sleep(.002)
            active -= 1
            if alias not in ablation_done:
                sent += 90
                ablation_done.add(alias)
            target = self.root/'results'/'.ablation_cache'/f'{alias}.requests.jsonl'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(''.join(json.dumps(dict(question_id=f'ab-{i}',condition='image',status='ok'))+'\n'
                                      for i in range(90)),'utf-8')
            events.append((alias,'ablation','end'))
            return [None]*1440,dict(model=alias,variant_records=1440)
        async def exercise_pipeline():
            nonlocal claude_main_started
            # Each asyncio.run gets its own event so resume is tested against
            # the same dependency, rather than an event left set by the first run.
            claude_main_started = asyncio.Event()
            await run_final.pipeline(models,Tracker(),Tracker(),parallel_models=True)
        with patch.object(run_final,'ROOT',self.root), patch.object(run_final,'OUT',self.out), \
             patch.object(run_final,'ordered_mini',return_value=questions), \
             patch.object(run_final,'vision_controls',return_value=controls), \
             patch.object(run_final,'model_config',side_effect=config), \
             patch.object(run_final,'evaluate_questions',side_effect=evaluate), \
             patch.object(run_final,'load_questions',return_value=[]), \
             patch.object(run_final,'load_scenes',return_value={}), \
             patch.object(run_final,'select_questions',side_effect=select), \
             patch.object(run_final,'build_ablation_plan',side_effect=plan), \
             patch.object(run_final,'run_ablation',side_effect=ablate), patch.object(run_final,'emit'):
            asyncio.run(exercise_pipeline())
            first_sent = sent
            self.assertGreaterEqual(peak,3)
            self.assertLess(events.index(('claude','image','start')),events.index(('kimi','smoke','end')))
            saved = json.loads((self.out/'status.json').read_text())
            self.assertEqual(set(saved),set(models))
            for alias in models:
                self.assertTrue(saved[alias]['smoke_passed'])
                self.assertEqual(set(saved[alias]['main']),set(run_final.CONDITIONS))
                self.assertEqual(saved[alias]['ablation']['variant_records'],1440)
                self.assertLess(events.index((alias,'smoke','end')),events.index((alias,'image','start')))
                self.assertLess(events.index((alias,'no_image','end')),events.index((alias,'ablation','start')))
            self.assertEqual(set(json.loads((self.out/'ablation_budget_estimates.json').read_text())),set(models))
            self.assertEqual(baseline_calls,list(run_final.CONDITIONS))
            asyncio.run(exercise_pipeline())
            self.assertEqual(sent,first_sent)
            self.assertEqual(baseline_calls,list(run_final.CONDITIONS)*2)

    def test_parallel_error_waits_for_other_models_to_settle(self):
        completed = []
        async def worker(alias):
            if alias == 'bad': raise ValueError('mock failure')
            await asyncio.sleep(.005)
            completed.append(alias)
        with self.assertRaisesRegex(RuntimeError,'bad: ValueError'):
            asyncio.run(run_final.dispatch_models(['bad','good'],worker,parallel=True))
        self.assertEqual(completed,['good'])


if __name__ == '__main__':
    unittest.main()
