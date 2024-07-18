import re
from pathlib import Path

# 一些特殊的token的名称
DEFAULT_BEGIN_TOKEN = "<cls>"
DEFAULT_END_TOKEN = "<end>"
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_UNK_TOKEN = "?"

class MolEncTokeniser:
    def __init__(
        self,
        vocab, # 整个词汇表
        chem_token_idxs,
        prog, # compiled extro tokens
        begin_token=DEFAULT_BEGIN_TOKEN,
        end_token=DEFAULT_END_TOKEN,
        pad_token=DEFAULT_PAD_TOKEN,
        unk_token=DEFAULT_UNK_TOKEN,
    ):
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.decode_vocab = {i: t for t, i in self.vocab.items()}
        self.chem_token_idxs = chem_token_idxs
        self.prog = prog

        self.begin_token = begin_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.unk_id = self.vocab[unk_token]
        self.unk_token_cnt = {}

    @staticmethod
    def from_vocab_file(
        vocab_path,
        regex,
        chem_tokens_start_idx,
        pad_token_idx=0,
        unk_token_idx=1,
        begin_token_idx=2,
        end_token_idx=3
    ):
        # 读取xc_work/bart_vocab.txt文件，得到的对象就是tokens。
        text = Path(vocab_path).read_text() # 创建Path对象，并读取文件内容
        tokens = text.split("\n")
        tokens = [t for t in tokens if t is not None and t != ""] # 除去None和""

        token_idxs = [pad_token_idx, unk_token_idx, begin_token_idx, end_token_idx]
        extra_tokens_idxs = range(max(token_idxs) + 1, chem_tokens_start_idx)
        extra_tokens = [tokens[idx] for idx in extra_tokens_idxs]
        prog = MolEncTokeniser._get_compiled_regex(regex, extra_tokens)
        pad_token = tokens[pad_token_idx]
        unk_token = tokens[unk_token_idx]
        begin_token = tokens[begin_token_idx]
        end_token = tokens[end_token_idx]
        chem_tokens_idxs = list(range(chem_tokens_start_idx, len(tokens)))
        tokeniser = MolEncTokeniser(
            tokens,
            chem_tokens_idxs,
            prog,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token
        )
        return tokeniser

    def __len__(self):
        return len(self.vocab)

    def tokenise(self, sents1, sents2=None, mask=False, pad=False):
        if sents2 is not None and len(sents1) != len(sents2):
            raise ValueError("Sentence 1 batch and sentence 2 batch must have the same number of elements")

        # 匹配所有的extra tokens
        tokens = self._regex_match(sents1)
        # m_token : 对SMIELS的tokenize。例如：[['C'], ['C'], ['['], ['C'], ['@'], ['@'], ['H'], [']'], ['C'], ['O']]
        # token_masks ：是否MASK，[Bool]。例如： [[False], [False], [False], [False], [False], [False], [False], [False], [False], [False]]
        m_tokens, token_masks = self._mask_tokens(tokens, empty_mask=not mask)

        sent_masks = None
        # 为extra tokens中所有的项，加上"^"和"&"
        tokens = [[self.begin_token] + ts + [self.end_token] for ts in tokens]
        # 为SMILES tokenize之后的所有项，加上"^"和"&"
        m_tokens = [[self.begin_token] + ts + [self.end_token] for ts in m_tokens]
        # 为token_masks之后的所有项，加上[False]和[False]，其实也代表"^"和"&"不MASK
        token_masks = [[False] + ts + [False] for ts in token_masks]
        sent_masks = [[0] + mask + [1] for mask in sent_masks] if sent_masks is not None else None

        output = {}

        if pad:
            tokens, orig_pad_masks = self._pad_seqs(tokens, self.pad_token)
            m_tokens, masked_pad_masks = self._pad_seqs(m_tokens, self.pad_token)
            token_masks, _ = self._pad_seqs(token_masks, False)
            sent_masks, _ = self._pad_seqs(sent_masks, False) if sent_masks is not None else (None, None)
            output["original_pad_masks"] = orig_pad_masks
            output["masked_pad_masks"] = masked_pad_masks

        output["original_tokens"] = tokens

        if mask:
            output["masked_tokens"] = m_tokens
            output["token_masks"] = token_masks

        if sent_masks is not None:
            output["sentence_masks"] = sent_masks

        return output

    def _regex_match(self, smiles):
        tokenised = []
        for smi in smiles:
            # 返回一个所有的字符串列表
            tokens = self.prog.findall(smi)
            tokenised.append(tokens)

        return tokenised

    @staticmethod
    def _get_compiled_regex(regex, extra_tokens):
        regex_string = r"("
        for token in extra_tokens:
            processed_token = token
            for special_character in "()[].|":
                # string.replace(old,new)：将old子串替换为new子串。
                processed_token = processed_token.replace(special_character, f"\\{special_character}") # 也就在special_character最前面加了一个\
            regex_string += processed_token + r"|"

        regex_string += regex + r"|"
        regex_string += r".)"
        return re.compile(regex_string)

    def _concat_sentences(self, tokens1, tokens2, sep):
        tokens = [ts1 + [sep] + ts2 for ts1, ts2 in zip(tokens1, tokens2)]
        sent_masks = [([0] * len(ts1)) + [0] + ([1] * len(ts2)) for ts1, ts2 in zip(tokens1, tokens2)]
        return tokens, sent_masks

    @staticmethod
    def _pad_seqs(seqs, pad_token):
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks
