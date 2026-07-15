## Self-Attention

attention is all you need

self-attention中b1-b4都看过了整个sequence，但可以并行计算(RNN不行）

### Embedding to a

embedding从x_{i}得到a_{i}

q: query(to match others)

k: key(to be matched)

v: value(information to be extracted)

### a_{n} to b_{n}

put every (query q) to do attention on every (key k)

get alpha

scaled Dot-Product **Attention**
![[Pasted image 20260408112911.png]]
add softmax layer(normalization) get alpha_head
![[Pasted image 20260408112930.png]]

get b1-b4 in the same way

## How self-attention do parallel
![[Pasted image 20260408113018.png]]

In scaled Dot-Product Attention

it is the same
![[Pasted image 20260408113059.png]]

What does self-attention layer does

- calculate Q、K、V
- get Attention A
- use softmax get A_head
- get O

## Multi-head self-attention

q, k, v will seperate into two q, k, v

multi-head can do what you like in different head

## Positional Encoding

- No position information in self-attention
    - sequence don’t have difference
- Original paper: each position has a unique positional vector e_{i}(not learned from data)
- In other words: each x_{i} appends a one-hot vector p_{i}

## Seq2Seq with Attention

core: replace all RNN with self-attention

## Transformer

If you can use seq2seq, you can use Transformer
![[Notion 2026-04-08 12.56.54.png]]