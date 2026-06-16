from transformers import PreTrainedTokenizerFast

def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        'skt/kogpt2-base-v2',
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>',
    )
    return tokenizer

tokenizer = load_tokenizer()
print(f'토크나이저 로드 완료  |  vocab size: {tokenizer.vocab_size:,}')




# from transformers import GPT2Tokenizer

# def load_tokenizer():
#     tokenizer = GPT2Tokenizer.from_pretrained(
#         "skt/kogpt2-base-v2",
#         bos_token="</s>",
#         eos_token="</s>",
#         unk_token="<unk>",
#         pad_token="<pad>",
#         mask_token="<mask>",
#     )
#     return tokenizer