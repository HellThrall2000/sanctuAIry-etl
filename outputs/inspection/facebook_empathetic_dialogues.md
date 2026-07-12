# Inspection Report: EmpatheticDialogues

**Hugging Face ID:** `facebook/empathetic_dialogues`  
**Inspection Date:** 2026-07-09 19:19 UTC  
**Splits:** `train`, `validation`, `test`

---

## Split: `train` — 76,673 rows

### Schema

| # | Column | Type | Nested |
|---|--------|------|--------|
| 1 | `conv_id` | `string` | No |
| 2 | `utterance_idx` | `int32` | No |
| 3 | `context` | `string` | No |
| 4 | `prompt` | `string` | No |
| 5 | `speaker_idx` | `int32` | No |
| 6 | `utterance` | `string` | No |
| 7 | `selfeval` | `string` | No |
| 8 | `tags` | `string` | No |

### Conversation Structure

| Metric | Value |
|--------|-------|
| Detection method | `id_grouped` |
| Grouping column | `conv_id` |
| Conversations | 17,844 |
| Avg turns per conversation | 4.3 |
| Median turns | 4.0 |
| Min turns | 1 |
| Max turns | 8 |

### Message Statistics

| Metric | Value |
|--------|-------|
| Columns analyzed | `conv_id`, `context`, `prompt`, `utterance`, `selfeval`, `tags` |
| Avg character length | 35.9 |
| Max character length | 117,507 |

### Data Quality

| Check | Column | Count |
|-------|--------|-------|
| Empty strings | `tags` | 75,975 |
| Extremely long (>5000ch) | `utterance` | 41 |
| Extremely short (<10ch) | `context` | 63,600 |
| Extremely short (<10ch) | `prompt` | 128 |
| Extremely short (<10ch) | `utterance` | 668 |
| Extremely short (<10ch) | `selfeval` | 5 |
| Extremely short (<10ch) | `tags` | 470 |

### Sample Records (5)

#### Sample 1

```json
{
  "conv_id": "hit:2012_conv:4024",
  "utterance_idx": 4,
  "context": "nostalgic",
  "prompt": "I went to watch my father play a softball game for the first time in many years. Growing up_comma_ I used to always watch his games.",
  "speaker_idx": 234,
  "utterance": "That's fantastic :) Nostalgia is an amazing thing",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 2

```json
{
  "conv_id": "hit:4134_conv:8268",
  "utterance_idx": 6,
  "context": "impressed",
  "prompt": "MY friend of 6 years was saving to buy a mercedes when I met her. She finally got it this year.",
  "speaker_idx": 244,
  "utterance": "I can. Its good she achieved her dream.",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 3

```json
{
  "conv_id": "hit:5728_conv:11456",
  "utterance_idx": 1,
  "context": "annoyed",
  "prompt": "This headache needs to go away.",
  "speaker_idx": 9,
  "utterance": "This headache needs to go away!",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 4

```json
{
  "conv_id": "hit:5855_conv:11711",
  "utterance_idx": 3,
  "context": "anticipating",
  "prompt": "My mom is getting remarried this weekend.",
  "speaker_idx": 220,
  "utterance": "We've actually been working really hard to convert their backyard into a fun venue! It's large and landscaped enough_comma_ it's giving them a lot more freedom with what they can acheive!",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 5

```json
{
  "conv_id": "hit:7486_conv:14973",
  "utterance_idx": 3,
  "context": "lonely",
  "prompt": "My best friend off to London . We bread and buttered together.  Eben though am happy for him _comma_ I am going to miss him a lot",
  "speaker_idx": 43,
  "utterance": "Yeah_comma_ we bread and buttered together from childhood. I think i am going to miss him a lot. I already fee like losing my soul and being alone",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

---

## Split: `validation` — 12,030 rows

### Schema

| # | Column | Type | Nested |
|---|--------|------|--------|
| 1 | `conv_id` | `string` | No |
| 2 | `utterance_idx` | `int32` | No |
| 3 | `context` | `string` | No |
| 4 | `prompt` | `string` | No |
| 5 | `speaker_idx` | `int32` | No |
| 6 | `utterance` | `string` | No |
| 7 | `selfeval` | `string` | No |
| 8 | `tags` | `string` | No |

### Conversation Structure

| Metric | Value |
|--------|-------|
| Detection method | `id_grouped` |
| Grouping column | `conv_id` |
| Conversations | 2,763 |
| Avg turns per conversation | 4.35 |
| Median turns | 4.0 |
| Min turns | 1 |
| Max turns | 8 |

### Message Statistics

| Metric | Value |
|--------|-------|
| Columns analyzed | `conv_id`, `context`, `prompt`, `utterance`, `selfeval`, `tags` |
| Avg character length | 37.1 |
| Max character length | 61,803 |

### Data Quality

| Check | Column | Count |
|-------|--------|-------|
| Empty strings | `selfeval` | 5 |
| Empty strings | `tags` | 11,952 |
| Extremely long (>5000ch) | `utterance` | 5 |
| Extremely short (<10ch) | `context` | 9,867 |
| Extremely short (<10ch) | `prompt` | 45 |
| Extremely short (<10ch) | `utterance` | 94 |
| Extremely short (<10ch) | `tags` | 54 |

### Sample Records (5)

#### Sample 1

```json
{
  "conv_id": "hit:4181_conv:8362",
  "utterance_idx": 1,
  "context": "sentimental",
  "prompt": "i felt nostalgic looking through old photos today",
  "speaker_idx": 44,
  "utterance": "i felt nostalgic looking through old photos today",
  "selfeval": "4|5|4_5|5|5",
  "tags": ""
}
```

#### Sample 2

```json
{
  "conv_id": "hit:8101_conv:16203",
  "utterance_idx": 3,
  "context": "sentimental",
  "prompt": "When I saw my mom after a long time I feel something",
  "speaker_idx": 508,
  "utterance": "Why! what is the reason",
  "selfeval": "5|5|5_1|1|1",
  "tags": ""
}
```

#### Sample 3

```json
{
  "conv_id": "hit:9039_conv:18078",
  "utterance_idx": 4,
  "context": "afraid",
  "prompt": "I was scared of the spider in my bathroom.",
  "speaker_idx": 463,
  "utterance": "how in the world did it get in your house? were you able to kill it?",
  "selfeval": "5|5|5_4|5|5",
  "tags": ""
}
```

#### Sample 4

```json
{
  "conv_id": "hit:10328_conv:20657",
  "utterance_idx": 2,
  "context": "sentimental",
  "prompt": "I was doing a big clear out recently and came accross my high school year book.  It has been many years since I left school and it was wonderul to find it again.  I spent hours going through it and looking at all the people long gone from my life.  Such a feeling of nostalgia and some sadness. Though it evoked happy memories too especially of prom night and our final year leaving party.  Some stories could be told about that!",
  "speaker_idx": 701,
  "utterance": "Did you guys write still write notes in each others' books in high school?",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 5

```json
{
  "conv_id": "hit:12205_conv:24411",
  "utterance_idx": 1,
  "context": "confident",
  "prompt": "I studied all night for my final exam",
  "speaker_idx": 547,
  "utterance": "I studied all night for my final exam\"",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

---

## Split: `test` — 10,943 rows

### Schema

| # | Column | Type | Nested |
|---|--------|------|--------|
| 1 | `conv_id` | `string` | No |
| 2 | `utterance_idx` | `int32` | No |
| 3 | `context` | `string` | No |
| 4 | `prompt` | `string` | No |
| 5 | `speaker_idx` | `int32` | No |
| 6 | `utterance` | `string` | No |
| 7 | `selfeval` | `string` | No |
| 8 | `tags` | `string` | No |

### Conversation Structure

| Metric | Value |
|--------|-------|
| Detection method | `id_grouped` |
| Grouping column | `conv_id` |
| Conversations | 2,542 |
| Avg turns per conversation | 4.3 |
| Median turns | 4.0 |
| Min turns | 1 |
| Max turns | 8 |

### Message Statistics

| Metric | Value |
|--------|-------|
| Columns analyzed | `conv_id`, `context`, `prompt`, `utterance`, `selfeval`, `tags` |
| Avg character length | 40.3 |
| Max character length | 72,319 |

### Data Quality

| Check | Column | Count |
|-------|--------|-------|
| Empty strings | `selfeval` | 4 |
| Empty strings | `tags` | 10,868 |
| Extremely long (>5000ch) | `utterance` | 4 |
| Extremely short (<10ch) | `context` | 8,944 |
| Extremely short (<10ch) | `prompt` | 18 |
| Extremely short (<10ch) | `utterance` | 71 |
| Extremely short (<10ch) | `tags` | 52 |

### Sample Records (5)

#### Sample 1

```json
{
  "conv_id": "hit:2090_conv:4181",
  "utterance_idx": 2,
  "context": "ashamed",
  "prompt": "I clogged up the toilet at a female friend's house by accident. I didn't tell them.",
  "speaker_idx": 75,
  "utterance": "Did you have a severe diarreia or something?",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 2

```json
{
  "conv_id": "hit:3534_conv:7068",
  "utterance_idx": 1,
  "context": "joyful",
  "prompt": "I feel happy when I am outside in nature.",
  "speaker_idx": 346,
  "utterance": "I feel happy when I am outside in nature",
  "selfeval": "3|5|4_5|5|5",
  "tags": ""
}
```

#### Sample 3

```json
{
  "conv_id": "hit:10647_conv:21294",
  "utterance_idx": 4,
  "context": "impressed",
  "prompt": "One of my friends managed to pay off their first house at the age of 25. ",
  "speaker_idx": 559,
  "utterance": "Understandable_comma_ i bet the had good budgeting skills and a nice job.",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

#### Sample 4

```json
{
  "conv_id": "hit:11772_conv:23545",
  "utterance_idx": 1,
  "context": "content",
  "prompt": "One time I got a week's vacation from work. I got to spend a lot of time watching my favorite shows",
  "speaker_idx": 630,
  "utterance": "I've got a week off from work. Time to sleep in and binge watch my favorite shows.",
  "selfeval": "5|5|5_5|4|5",
  "tags": ""
}
```

#### Sample 5

```json
{
  "conv_id": "hit:12423_conv:24847",
  "utterance_idx": 2,
  "context": "anxious",
  "prompt": "I have a big test on Monday. I am so nervous_comma_ I haven't been able to sleep at all.",
  "speaker_idx": 375,
  "utterance": "What is the test on?",
  "selfeval": "5|5|5_5|5|5",
  "tags": ""
}
```

