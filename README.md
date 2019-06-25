# bert-encoder

# Usage
```python
from bert_encoder import BertEncoder
be = BertEncoder("/path/to/pretrain_bert_model")
be.encode(["你好", "你在哪"])
```

output

```
[array([[-0.43364248,  0.41179657, -0.32683408, ...,  0.2388876 ,
         0.2813817 , -0.12212703],
       [-0.3853165 ,  0.30760592, -0.3487961 , ..., -0.02057578,
        -0.21075164, -0.4568899 ]], dtype=float32), array([[-0.43364248,  0.41179657, -0.32683408, ...,  0.2388876 ,
         0.2813817 , -0.12212703],
       [-0.1804257 ,  0.7799143 , -0.19864915, ..., -0.26258343,
         0.01706939, -0.6424839 ],
       [-0.45424426,  0.5611929 ,  0.1049192 , ...,  0.31221464,
         0.19015062, -0.10684527]], dtype=float32)]

```
# Environment
```
python 3.5+
```

# Installation
```bash
cd bert-encoder
python setup install
```