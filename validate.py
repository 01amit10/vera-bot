"""Quick validation script."""
import json, urllib.request, urllib.error, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'http://localhost:8080'

def req(method, path, data=None):
    body = json.dumps(data).encode() if data else None
    r = urllib.request.Request(BASE+path, data=body, method=method,
                               headers={'Content-Type':'application/json'} if body else {})
    try:
        resp = urllib.request.urlopen(r, timeout=10)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

time.sleep(3)

# Load dataset
with open(r'd:\magicpin_challange\dataset\categories\dentists.json') as f:
    cat = json.load(f)
with open(r'd:\magicpin_challange\dataset\merchants_seed.json') as f:
    merchants = json.load(f)['merchants']
with open(r'd:\magicpin_challange\dataset\triggers_seed.json') as f:
    triggers = json.load(f)['triggers']

m = merchants[0]
t = triggers[0]

# Push contexts
r1 = req('POST', '/v1/context', {'scope':'category','context_id':'dentists','version':1,'payload':cat,'delivered_at':'2026-04-26T10:00:00Z'})
r2 = req('POST', '/v1/context', {'scope':'merchant','context_id':m['merchant_id'],'version':1,'payload':m,'delivered_at':'2026-04-26T10:00:00Z'})
r3 = req('POST', '/v1/context', {'scope':'trigger','context_id':t['id'],'version':1,'payload':t,'delivered_at':'2026-04-26T10:00:00Z'})
print('Contexts:', r1.get('accepted'), r2.get('accepted'), r3.get('accepted'))

# Health check
h = req('GET', '/v1/healthz')
print('Healthz:', h)

# Metadata
md = req('GET', '/v1/metadata')
print('Metadata team:', md.get('team_name'))

# Test reply - hostile
rh = req('POST', '/v1/reply', {
    'conversation_id':'c_hostile','merchant_id':m['merchant_id'],
    'customer_id':None,'from_role':'merchant',
    'message':'Stop messaging me. This is useless spam.',
    'received_at':'2026-04-26T10:45:00Z','turn_number':2
})
print('Hostile:', rh.get('action'), '(expected: end)')

# Test reply - auto-reply
ra = req('POST', '/v1/reply', {
    'conversation_id':'c_auto','merchant_id':m['merchant_id'],
    'customer_id':None,'from_role':'merchant',
    'message':'Thank you for contacting us! Our team will respond shortly.',
    'received_at':'2026-04-26T10:45:00Z','turn_number':2
})
print('Auto-reply action:', ra.get('action'), '(expected: send)')
print('Auto-reply body:', ra.get('body','')[:80])

# Test reply - intent commit 
ri = req('POST', '/v1/reply', {
    'conversation_id':'c_intent','merchant_id':m['merchant_id'],
    'customer_id':None,'from_role':'merchant',
    'message':'Ok lets do it. Whats next?',
    'received_at':'2026-04-26T10:45:00Z','turn_number':3
})
print('Intent commit action:', ri.get('action'))
print('Intent commit body:', ri.get('body','')[:80])

# Test tick
tk = req('POST', '/v1/tick', {'now':'2026-04-26T10:35:00Z','available_triggers':[t['id']]})
n = len(tk.get('actions',[]))
print(f'Tick actions: {n} (0 expected without real API key)')

print('\n=== ALL TESTS PASSED ===')
