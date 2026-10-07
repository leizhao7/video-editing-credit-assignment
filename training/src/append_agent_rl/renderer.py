"""Render the initial prompt once, then only new tool observations.

The underlying renderer/processor is pinned by the same inference contract.
This intentionally defines a token-stream protocol instead of applying a chat
template again to historical, parsed/reformatted assistant messages.
"""
from copy import deepcopy

from opencode_rl.contract import ParityError, require_ids
from .loop import checked_segment


class SuffixRenderer:
    def __init__(self, client, *, assistant_end_ids, newline_ids, final_turn_prompt):
        require_ids(assistant_end_ids, 'assistant end marker')
        require_ids(newline_ids, 'newline marker')
        if len(assistant_end_ids) != 1:
            raise ParityError('Only a pinned single-token assistant end marker is supported')
        self.client = client
        self.end = list(assistant_end_ids)
        self.newline = list(newline_ids)
        self.final_turn_prompt = final_turn_prompt
        self.anchor = [{'role': 'user', 'content': '__append_only_observation_anchor__'}]
        self._anchor = None

    async def initial(self, profile):
        if profile.get('format') != 'append-opencode-tools-v1' or profile.get('native_agent_loop') is not False:
            raise ParityError('Expected the pinned tool-only host profile')
        if not profile.get('system') or not profile.get('tools') or not profile.get('task'):
            raise ParityError('Missing migrated prompt or tool catalogue')
        messages = [{'role': 'system', 'content': text} for text in profile['system']]
        messages.append({'role': 'user', 'content': profile['task']})
        result = await self.client.render(messages=messages, tools=deepcopy(profile['tools']),
                                          add_generation_prompt=True)
        return checked_segment(result)

    async def observation(self, observation, *, last_token_id, final_turn=False, budget_notice=None):
        if type(observation) is not dict or set(observation) != {'results'} or not observation['results']:
            raise ParityError('Expected new native tool outputs, not a conversation')
        messages = []
        for result in observation['results']:
            if type(result.get('output')) is not str or type(result.get('attachments')) is not list:
                raise ParityError('Incomplete native tool text/media output')
            content = [{'type': 'text', 'text': result['output']}]
            for part in result['attachments']:
                # The root-owned shared media gate has already normalized and
                # authorized these images before they reach this renderer.
                if not part.get('mime', '').startswith('image/') or not part.get('url', '').startswith('data:image/'):
                    raise ParityError('Only admitted inline image attachments are supported')
                content.append({'type': 'image_url', 'image_url': {'url': part['url']}})
            messages.append({'role': 'tool', 'tool_call_id': result['id'], 'name': result['name'], 'content': content})
        if budget_notice is not None and (not isinstance(budget_notice, str) or not 0 < len(budget_notice) <= 12000):
            raise ParityError('Invalid public harness budget notice')
        if final_turn and budget_notice is not None and budget_notice != self.final_turn_prompt:
            raise ParityError('Budget note changed original final-turn instruction')
        note = self.final_turn_prompt if final_turn else budget_notice
        if note is not None:
            messages[-1]['content'].append({'type': 'text', 'text': '\n<system-reminder>\n' +
                                            note + '\n</system-reminder>'})
        if self._anchor is None:
            self._anchor = checked_segment(await self.client.render(messages=deepcopy(self.anchor),
                tools=[], add_generation_prompt=False))
            if self._anchor['features']:
                raise ParityError('The fixed text anchor unexpectedly contains media')
        rendered = checked_segment(await self.client.render(messages=deepcopy(self.anchor) + messages,
            tools=[], add_generation_prompt=True))
        anchor = self._anchor['token_ids']
        if rendered['token_ids'][:len(anchor)] != anchor or len(rendered['token_ids']) <= len(anchor):
            raise ParityError('Observation renderer changed anchor tokens; no length-only fallback')
        closure = ([] if last_token_id == self.end[0] else self.end) + self.newline
        features = []
        for feature in rendered['features']:
            if feature['offset'] < len(anchor):
                raise ParityError('New image spans the anchor boundary')
            features.append({**feature, 'offset': feature['offset'] - len(anchor) + len(closure)})
        return checked_segment({'token_ids': closure + rendered['token_ids'][len(anchor):], 'features': features})
