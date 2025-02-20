import re

def extract_hallucination_spans(text):
    """
    从文本中提取幻觉片段,返回每个片段的起始位置、结束位置、文本内容和confidence值(如果有)
    
    Args:
        text (str): 输入文本
        
    Returns:
        list: 包含字典的列表,每个字典包含:
            - start: 起始位置(按单词计数)
            - end: 结束位置(按单词计数)
            - text: 幻觉文本内容
            - confidence: confidence值(如果有)
    """
    # 定义两种模式的正则表达式
    pattern_with_conf = r'<hallucination confidence=(\d+)>(.*?)</hallucination>'
    pattern_without_conf = r'<hallucination>(.*?)</hallucination>'
    
    # 存储所有单词的列表
    words = text.split()
    
    # 存储结果
    spans = []
    
    # 首先处理带confidence的情况
    for match in re.finditer(pattern_with_conf, text):
        confidence = int(match.group(1))
        hallucination_text = match.group(2)
        
        # 计算这段幻觉文本在原文中的起始和结束位置(按单词)
        start_char = match.start()
        prefix_text = text[:start_char]
        start_word = len(prefix_text.split())
        
        # 计算结束位置
        hallucination_words = hallucination_text.split()
        end_word = start_word + len(hallucination_words)
        
        spans.append({
            'start': start_word,
            'end': end_word,
            'text': hallucination_text,
            'confidence': confidence
        })
    
    # 处理不带confidence的情况
    for match in re.finditer(pattern_without_conf, text):
        hallucination_text = match.group(1)
        
        # 计算这段幻觉文本在原文中的起始和结束位置(按单词)
        start_char = match.start()
        prefix_text = text[:start_char]
        start_word = len(prefix_text.split())
        
        # 计算结束位置
        hallucination_words = hallucination_text.split()
        end_word = start_word + len(hallucination_words)
        
        spans.append({
            'start': start_word,
            'end': end_word,
            'text': hallucination_text,
            'confidence': None
        })
    
    # 按起始位置排序
    spans.sort(key=lambda x: x['start'])
    
    return spans

# 测试代码
if __name__ == "__main__":
    # 测试带confidence的情况
    text1 = 'Solution: <hallucination confidence=10>cos30° = frac{2.5}{AB}</hallucination>, therefore <hallucination confidence=9>AB = √{3}</hallucination>. Thus, <hallucination confidence=8>the answer is C</hallucination>.  \nAnswer: C'
    spans1 = extract_hallucination_spans(text1)
    print("Test 1 results:")
    for span in spans1:
        print(span)
    
    # 测试不带confidence的情况
    text2 = 'Solution: cos30° = <hallucination>frac {2.5}{AB}</hallucination>, therefore AB = <hallucination>√{3}</hallucination>. Thus, the answer is C.  \nAnswer: C'
    spans2 = extract_hallucination_spans(text2)
    print("\nTest 2 results:")
    for span in spans2:
        print(span) 