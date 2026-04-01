import requests
def make_api_request(api_path, method='GET', data=None):
    url = f'https://app.base44.com/api/{api_path}'
    headers = {
        'api_key': '82e4ca558e8546f89859b4a3dba1e1cf',
        'Content-Type': 'application/json'
    }
    if method.upper() == 'GET':
        response = requests.request(method, url, headers=headers, params=data)
    else:
        response = requests.request(method, url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()
entities = make_api_request(f'apps/696fe84f14c617992088dd7d/entities/BotStatus')
print(entities)
def update_entity(entity_id, update_data):
    response = requests.put(
        f'https://app.base44.com/api/apps/696fe84f14c617992088dd7d/entities/BotStatus/{entity_id}',
        headers={
            'api_key': '82e4ca558e8546f89859b4a3dba1e1cf',
            'Content-Type': 'application/json'
        },
        json=update_data
    )
    response.raise_for_status()
    return response.json()
