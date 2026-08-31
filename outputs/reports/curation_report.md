# Training Curation Report

Documents how the deduplicated corpus was cleaned, filtered and
rebalanced into the final supervised fine-tuning splits.

## Summary

- **Training conversations**: 5144
- **Validation conversations**: 271
- **Random seed**: 42

## Source Funnel

| Source | Deduplicated | After Quality | After Rebalancing | Final Share |
| :--- | ---: | ---: | ---: | ---: |
| `Amod/mental_health_counseling_conversations` | 2025 | 2022 | 2022 | 37.34% |
| `facebook/empathetic_dialogues` | 17780 | 14192 | 2500 | 46.17% |
| `thu-coai/esconv` | 910 | 893 | 893 | 16.49% |

## Text Repairs

- **Messages inspected**: 107307
- **Messages rewritten**: 31594

| Rule | Replacements |
| :--- | ---: |
| `_comma_` | 26031 |
| `missing_space` | 3643 |
| `url` | 134 |

## Quality Filtering

- **Conversations inspected**: 20715
- **Conversations retained**: 17107

| Drop Reason | Conversations |
| :--- | ---: |
| `too_few_messages` | 3502 |
| `mostly_short_responses` | 85 |
| `assistant_echoes_user` | 21 |
