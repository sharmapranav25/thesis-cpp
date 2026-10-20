import copy,json
import experiment as e
recipes=e.A.read(e.OUT/'City/fit_recipes.json')[:3]
checks=0
for recipe in recipes:
    original=copy.deepcopy(recipe)
    for key in e.ALL:
        changed=e.edit(recipe,{key:'off'})
        assert changed['text_spec']['text']==recipe['text_spec']['text']
        assert changed['seed']==recipe['seed'] and changed['size']==recipe['size']
        kind,h=key.split('.')
        if kind=='effect':
            assert changed['text_spec']==recipe['text_spec']
            assert changed['effects_spec']['background']==recipe['effects_spec']['background']
            assert changed['effects_spec']['effects']==[v for v in recipe['effects_spec']['effects'] if e.P.E.REGISTRY[v['name']]['head']!=h]
        else:assert changed['effects_spec']==recipe['effects_spec']
        assert recipe==original
        checks+=1
    for h in e.FONT:
        assert e.edit(recipe,{'font.'+h:recipe['description']['font'][h]})['text_spec']==recipe['text_spec']
        checks+=1
result=dict(passed=True,cases=checks,checks=['Strings, seeds and sizes preserved','Input recipes not mutated','Effect removals preserve text and other effects','Text/font removals preserve effects','Unchanged font targets are exact no-ops'])
e.dump(e.OUT/'intervention_checks.json',result)
print(json.dumps(result))
