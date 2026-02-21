## 原始baseline

LLM：Llama-3.1-8B-Instruct

### WebQSP


```shell
Hit: 84.76658476658477
Hit@1: 81.01965601965603
Macro F1: 68.91269292839273
Macro Precision: 76.90711421534208
Macro Recall: 70.10918405274069
Exact Match: 48.34152334152334
Totally Wrong: 15.601965601965603
Hal Score: 81.56939091597324
```

### CWQ

```shell
Hit: 57.66071934296233
Hit@1: 52.05324270744831
Macro F1: 47.45335548599189
Macro Precision: 49.95532716798648
Macro Recall: 50.526317106849355
Exact Match: 37.49645992636647
Totally Wrong: 43.58538657604078
Hal Score: 63.10863864512125
```

## DDE拼接接入节点 + SEGA + node BCE

LLM：Llama-3.1-8B-Instruct

全局重排（alpha=0.5）

### WebQSP


```shell
Hit: 80.83538083538083
Hit@1: 76.84275184275184
Macro F1: 64.7854490937655
Macro Precision: 75.10657542343202
Macro Recall: 63.40056515710543
Exact Match: 43.980343980343974
Totally Wrong: 19.77886977886978
Hal Score: 81.21548146896676
```

### CWQ

```shell

```

---

LLM：Llama-3.1-8B-Instruct

锁Top50, 插入71-100

### WebQSP

```shell

```