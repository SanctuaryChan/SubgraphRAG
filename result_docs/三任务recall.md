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
调整l1-l4参数

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

固定lambda参数+锁Top50 替85-100 MaxPool

```shell
(srag1-2) ➜  retrieve git:(dev) ✗ python inference_stage2.py \
      -p webqsp_Feb13-13:49:52/cpt.pth \
      --stage2_path webqsp_Feb13-13:49:52/stage2_cpt.pth \
      -d webqsp \
      --inject_strategy structure \
      --topic_hop 1 \
      --lambda1 0.8 --lambda2 0.1 --lambda3 0.0 --lambda4 0.8 \
      --preserve_mid_end 85 --replace_start 86 --candidate_x 15 \
      --output_path webqsp_Feb13-13:49:52/retrieval_result_stage2_structure_strong_86-100.pth

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:00<00:00, 3179.87it/s]# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [04:40<00:00,  5.85it/s]Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_structure_strong_86-100.pth
Local rerank stats: applied=1631, fallback=7, partial=0
Structure inject stats: samples_with_pool=1631, pool_near_topic_edges=601077, pool_bridge_edges=244349
(srag1-2) ➜  retrieve git:(dev) ✗ 
(srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \
    -p webqsp_Feb13-13:49:52/retrieval_result_stage2_structure_strong_86-100.pth \ 
    --k_list 50,100,200,400,500
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.946                        0.880              0.868
200       0.973                        0.930              0.915
400       0.985                        0.962              0.950
500       0.987                        0.969              0.961
```

## G0-G4 小改DDE+loss

```shell
  (srag1-2) ➜  retrieve git:(dev) ✗ cat G0_webqsp_Feb21-12:16:39/eval_stage1.txt
    K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
   50       0.896                        0.814              0.785
  100       0.942                        0.873              0.842
  200       0.973                        0.922              0.899
  400       0.989                        0.963              0.953
  500       0.990                        0.970              0.965
  (srag1-2) ➜  retrieve git:(dev) ✗ cat G1_webqsp_Feb21-12:22:37/eval_stage1.txt
    K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
   50       0.909                        0.833              0.807
  100       0.944                        0.887              0.866
  200       0.971                        0.931              0.907
  400       0.986                        0.964              0.951
  500       0.990                        0.974              0.961
  (srag1-2) ➜  retrieve git:(dev) ✗ cat G2_webqsp_Feb21-12:32:30/eval_stage1.txt
    K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
   50       0.894                        0.827              0.810
  100       0.939                        0.889              0.863
  200       0.970                        0.934              0.910
  400       0.987                        0.965              0.947
  500       0.990                        0.971              0.958
  (srag1-2) ➜  retrieve git:(dev) ✗ cat G3_webqsp_Feb21-12:37:09/eval_stage1.txt
    K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
   50       0.910                        0.827              0.809
  100       0.955                        0.889              0.870
  200       0.978                        0.931              0.915
  400       0.989                        0.966              0.956
  500       0.991                        0.973              0.963
  (srag1-2) ➜  retrieve git:(dev) ✗ cat G4_webqsp_Feb21-12:43:11/eval_stage1.txt
    K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
   50       0.910                        0.836              0.810
  100       0.953                        0.896              0.869
  200       0.974                        0.938              0.915
  400       0.986                        0.970              0.955
  500       0.989                        0.976              0.962
```