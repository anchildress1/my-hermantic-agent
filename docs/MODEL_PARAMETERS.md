# Hermes-4-14B Model Parameters

Configuration guide for the Hermes-4-14B model parameters in `config/template.yaml`.

## About Hermes-4

> [!WARNING]
> **This Is Hermes, Not a Hall Monitor**
> 
> ⚠️ Hermes ships without the usual corporate-grade guardrails. He's a hybrid reasoning model with tool access and an attitude, and he will absolutely follow your instructions even when you probably shouldn't have written them. Before you grab this code and run, go read the [model card](https://huggingface.co/NousResearch/Hermes-4-14B) to understand what Hermes actually is and what he is not.

Hermes-4-14B is a powerful conversational model by NousResearch, designed for direct, unfiltered interactions. It purposefully lacks the guardrails commonly seen in other models.

**Model Card:** https://huggingface.co/NousResearch/Hermes-4-14B

---

## Parameter Reference

### temperature: 0.85

Controls randomness and creativity (0.0 to 2.0)

- **Lower (0.1-0.5)**: Focused, deterministic, predictable
- **Medium (0.6-0.9)**: Balanced creativity and coherence
- **Higher (1.0-2.0)**: Creative, unpredictable, experimental

**Current:** 0.85 - Good balance for conversational assistant

### top_p: 0.92

Nucleus sampling threshold

- Samples from top tokens until cumulative probability reaches this value
- 0.92 = considers top 92% most likely tokens
- Higher = more variety, lower = more focused

### top_k: 50

Limits sampling to K most likely tokens

- Prevents completely random outputs
- 50 = considers 50 most probable next words
- Balances variety with coherence

### num_predict: 4096

Maximum tokens per response

- 4096 tokens ≈ 3000 words
- Adjust based on desired response length

### repeat_penalty: 1.25

Penalizes repetition (1.0 = no penalty)

- 1.25 = moderate penalty
- Prevents annoying repetition without sounding unnatural

### repeat_last_n: 128

Look-back window for repetition detection

- Checks last 128 tokens (~100 words) for patterns
- Prevents getting stuck in loops

### presence_penalty: 0.6

Penalizes tokens that appeared in conversation

- Encourages diverse vocabulary
- 0.6 = moderate diversity boost

### frequency_penalty: 0.7

Penalizes frequently used tokens

- Reduces repetitive phrasing
- 0.7 = moderate penalty

### mirostat: 0

Adaptive sampling mode

- **0**: Disabled (using standard sampling)
- **1/2**: Enabled (dynamic quality adjustment)

**Current:** 0 - Standard sampling for predictable behavior

### num_ctx: 8192

Context window size in tokens

- 8192 = ~6000 words of context
- Larger = more memory, slower processing
- Agent keeps 75% for history, 25% for generation

---

## Tuning Tips

### For More Creative Responses

```yaml
temperature: 1.0
top_p: 0.95
mirostat: 2
mirostat_tau: 9.0
```

### For More Focused Responses

```yaml
temperature: 0.5
top_p: 0.85
top_k: 30
```

### For Longer Context

```yaml
num_ctx: 16384  # Requires more RAM
```

### To Reduce Repetition

```yaml
repeat_penalty: 1.5
presence_penalty: 0.8
frequency_penalty: 0.8
```

---

## Model Comparison

| Model | Size | Context | Speed | Quality |
|-------|------|---------|-------|---------|
| Hermes-4-14B | 14B | 8K | Medium | High |
| Llama-3.2-3B | 3B | 8K | Fast | Medium |
| Mistral-7B | 7B | 8K | Fast | High |
| Llama-3.1-70B | 70B | 128K | Slow | Very High |

---

## Resources

- **Model Card**: [[https://huggingface.co/NousResearch/Hermes-4-14B]]
- **Ollama Docs**: [[https://ollama.ai/library]]
- **Parameter Guide**: [[https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
