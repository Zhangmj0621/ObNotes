## 环境准备

下载所需的local dense retriever

```bash
# Flat index slow but accurate
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
conda activate retriever

# HNSW64
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```

下载依赖，版本必须对其，不然安装时会报依赖错误，且两个包只能安装一个（不能共存）

- faiss-gpu==1.8.0
- faiss-cpu==1.7.4

启动local dense retriever

```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```

## 启动训练

在verl中可直接使用examples/sglang_multiturn/search_r1_like/下的脚本进行搜索工具训练