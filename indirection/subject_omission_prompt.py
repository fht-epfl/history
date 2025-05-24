SYSTEM_PROMPT = """你是一個瞭解朱天心眷村文學的语言分析助手。現在有一段文字輸入，請你找出其中省略主語的短句，並返回填補主語后的句子，以及省略的主語。

**要求**：輸出必須是JSON格式的結構化數據，包括：
- `"original"`：直接複製原文句子。
- `"filled"`：填補主語后的句子。
- `"subject"`：省略的主語。
- `"reason"`：（可選）簡要說明為何該詞屬於上述分類。

請嚴格按照JSON數組的格式輸出所有標註結果，不要輸出除JSON之外的多餘內容。
"""

EXAMPLE_ONE = """的夜晚，你和Ａ躺在一張木床上，你還記得月光透過窗上的藤花、窗紗、連光帶影落在你們身上，前文忘了，只記得自己說：「反正將來我是不結婚的。」Ａ黑裡笑起來：「那×××不慘了。」×××是那時正勤寫信給你的男校同年級男生，一張大鼻大眼溫和的臉浮在你眼前，半天，Ａ說：「不知道同性戀好不好玩。」你沒回答，可能白天玩得太瘋了，沒再來得及交換一句話就沉沉睡去，貓咪打呼一般，兩具十七歲年輕的身體。"""
RESPONSE_ONE = """
[
{
"original": "前文忘了",
"filled": "我前文忘了",
"subject": "我",
"reason": "這句話的主語省略了，根據上下文可以推斷出是「我」。"
}
]
"""

EXAMPLE_TWO = """一定是你從無耐心聽人說夢、看人寫夢，以致失去向自己、向別人說夢的權利和習慣，更仿佛，隨身養了一頭食夢貘，總在太陽逐漸恢復溫暖之際，吃光你所有的夢。"""
RESPONSE_TWO = """
[
{
"original": "隨身養了一頭食夢貘",
"filled": "你隨身養了一頭食夢貘",
"subject": "你"
}
]
"""

EXAMPLE_THREE = """就會悚然驚懼，而非你們大多數的熱血沸騰當下想到陸皓東黃花崗……，同樣十幾歲的年紀，她們是如何做到的？以至在日後的啟蒙成長和獨立自主人格的養成上，省了好大一段冤枉路。二十年後政治正確的寫作者也許不難替她們安排一兩位二二八受難親族、或耕者有其田政策下被合法掠奪過家財的、或在牯嶺街買到《自由中國》或《大學雜誌》並因此啟蒙的，不然就有個替康寧祥郭雨"""
RESPONSE_THREE = """
[
{
"original": "以至在日後的啟蒙成長和獨立自主人格的養成上，省了好大一段冤枉路",
"filled": "以至她們在日後的啟蒙成長和獨立自主人格的養成上，省了好大一段冤枉路",
"subject": "她們",
"reason": "此句接續前文的「她們」，是作者表達艷羡的對象。"
},
{
"original": "或耕者有其田政策下被合法掠奪過家財的",
"filled": "或耕者有其田政策下被國民黨合法掠奪過家財的",
"subject": "國民黨"，
"reason": "根據上下文可以推斷此句省略了掠奪家財的施害者即國民黨，這是作者曾經認同但現在卻不願提及的政權"
}
]
"""

GENERATION_PROMPT = """{text}"""

default = [
    {"role": "user", "content": EXAMPLE_ONE},
    {"role": "assistant", "content": RESPONSE_ONE},
    {"role": "user", "content": EXAMPLE_TWO},
    {"role": "assistant", "content": RESPONSE_TWO},
    {"role": "user", "content": EXAMPLE_THREE},
    {"role": "assistant", "content": RESPONSE_THREE}
]


def prompt_formatter(text):

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *default,
        {"role": "user", "content": text},
    ]
    
    return prompt