Need you to prepare architecture markdown called spotlights_architecture.md in ./docs/architecture.


Following are the blocks:

0. SpotlightsManager - manages all other blocks

1. ModulesExtractor - see @docs/modules_extractor.md @src/spotlights_engine/schemas/modules.py @scripts/run_kv_offload.py

2. candidate_discovery - already implemented in src/spotlights_engine/candidate_discovery

3. module_deep_research

Input:
- prompt - string (it will be built from repo_info and module info from @src/spotlights_engine/schemas/modules.py )
- max_output_results - default 10


Output:

list of most relevant findings to the module:

fields:
- finding_id
- title
- url
- short description


4. finding_to_candidates_mapper

input:
- findings from module_deep_research output
- list of candidates: output from candidate_discovery

output:
- if fill each candiate with the related finding.

remark: @src/spotlights_engine/schemas/candidate.py : need to add findings field 

- This module loop on the findings and for each finiding start a claude session where the input is finding + list of candidates and it return the relevant candidates.

5. proposal_from_finding_creator

Input:
- candidates (with the findings from finding_to_candidates_mapper)

it fill the candidates with a proposal from each findings.
it can decide that a finding doesn't contain info from create a proposal.

add to candiate field called deep_research_proposals array of deep_research_proposal

The fields in deep_research_proposal are:
- title 
- detailed_description
- finding_id

6. agent_proposals

Input:
- candidates (output from proposal_from_finding_creator)
 

add to candidate field called agent_proposals array of agent_proposal

The fields in deep_research_proposal are:
- title 
- detailed_description
- agent_name

fill proposals not covered in deep_research_proposals