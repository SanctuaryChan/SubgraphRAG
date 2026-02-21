## 原始baseline（alpha=1.0）

```shell
Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_alpha1.0.pth
(srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \
    -p webqsp_Feb13-13:49:52/retrieval_result_stage2_alpha1.0.pth \
    --k_list 50,100,200,400,500 | tee webqsp_Feb13-13:49:52/eval_stage2_alpha1.0.txt
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.946                        0.880              0.859
200       0.972                        0.928              0.909
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
```


## 锁Top50 替换71-100 MaxPool

```shell
(srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \                                   
    -p webqsp_Feb13-13:49:52/retrieval_result_stage2_insert71-100.pth \
    --k_list 50,100,200,400,500
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.942                        0.875              0.859
200       0.973                        0.929              0.915
400       0.985                        0.962              0.949
500       0.987                        0.969              0.961
```

## 锁Top50 替81-100 MaxPool

```shell
# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [04:59<00:00,  5.48it/s]Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_local_r81-100_x20.pth
Local rerank stats: applied=1631, fallback=7, partial=0
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.944                        0.878              0.865
200       0.973                        0.930              0.916
400       0.985                        0.962              0.950
500       0.987                        0.969              0.961
```

## 锁Top50 替86-100 MaxPool

```shell
# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [04:54<00:00,  5.56it/s]Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_local_r86-100_x15.pth
Local rerank stats: applied=1631, fallback=7, partial=0
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.945                        0.879              0.866
200       0.973                        0.929              0.914
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
```

## 锁Top50 替91-100 MaxPool

```shell
# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [04:59<00:00,  5.47it/s]Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_local_r91-100_x10.pth
Local rerank stats: applied=1631, fallback=7, partial=0
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.947                        0.879              0.865
200       0.973                        0.929              0.913
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
```


## 锁Top50 替91-100 MaxPool + 使用结构感知注入分重排

```shell
(srag1-2) ➜  retrieve git:(dev) ✗ python inference_stage2.py \
    -p webqsp_Feb13-13:49:52/cpt.pth \
    --stage2_path webqsp_Feb13-13:49:52/stage2_cpt.pth \
    -d webqsp \
    --max_K 500 \
    --node_top_m 50 \
    --alpha 0.9 \
    --local_rerank \
    --lock_top_n 50 \
    --inject_strategy structure \
    --topic_hop 2 \
    --bridge_top_m 50 \
    --lambda1 1.0 \
    --lambda2 0.2 \
    --lambda3 0.25 \
    --lambda4 0.35 \
    --output_path webqsp_Feb13-13:49:52/retrieval_result_stage2_structure.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:00<00:00, 3091.55it/s]# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [05:02<00:00,  5.42it/s]Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_structure.pth
Local rerank stats: applied=1631, fallback=7, partial=0
Structure inject stats: samples_with_pool=1631, pool_near_topic_edges=647581, pool_bridge_edges=425862
(srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \
    -p webqsp_Feb13-13:49:52/retrieval_result_stage2_structure.pth \
    --k_list 50,100,200,400,500 \
    | tee webqsp_Feb13-13:49:52/eval_stage2_structure.txt
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.947                        0.879              0.866
200       0.973                        0.928              0.913
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
```

```shell
(srag1-2) ➜  retrieve git:(dev) ✗ python inference_stage2.py \
      -p webqsp_Feb13-13:49:52/cpt.pth \
      --stage2_path webqsp_Feb13-13:49:52/stage2_cpt.pth \
      -d webqsp \
      --inject_strategy structure \
      --topic_hop 1 \
      --lambda1 0.8 --lambda2 0.1 --lambda3 0.0 --lambda4 0.8 \
      --output_path webqsp_Feb13-13:49:52/retrieval_result_stage2_structure_strong.pth

  100%|
  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
  ██████████| 1639/1639 [00:00<00:00, 3161.65it/s]# skipped samples: 0
  # relevant triples | median: 4 | mean: 20 | max: 699
  100%|
  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
  ████████████| 1639/1639 [05:00<00:00,  5.46it/s]Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_structure_strong.pth
  Local rerank stats: applied=1631, fallback=7, partial=0
  Structure inject stats: samples_with_pool=1631, pool_near_topic_edges=601077, pool_bridge_edges=244350
  (srag1-2) ➜  retrieve git:(dev) ✗
  (srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \
      -p webqsp_Feb13-13:49:52/retrieval_result_stage2_structure_strong.pth \
      --k_list 50,100,200,400,500 \
      | tee webqsp_Feb13-13:49:52/eval_stage2_structure_strong.txt
    K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
   50       0.905                        0.827              0.800
  100       0.947                        0.881              0.866
  200       0.973                        0.930              0.914
  400       0.985                        0.962              0.949
  500       0.987                        0.969              0.961
```