from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str  # 数据集名称
    data_path: str  # 数据文件路径
    output_path: str  # 输出文件路径
    system_message: str  # 系统提示词
    user_prompt_template: str  # 用户提示词模板

# 系统提示词
SYSTEM_MESSAGE = """You are a precise hallucination detector for multimodal large language models. Your task is to:
1. Carefully examine the image
2. Read the model's response of the query related to image
3. First provide your analysis of hallucinations in <Analyze>...</Analyze> tags, including:
   - List each hallucinated detail
   - Explain why it's a hallucination
4. Then in <pre_output>...</pre_output> tags:
   - Simply output the original response with <hallucination> tags added
   - Add <hallucination> tags to EACH specific word or phrase that contains hallucinations
   - Be as specific and granular as possible in your tagging
   - Do not include any analysis or explanatory text
   - If no hallucinations, output the original text unchanged

Example:
<Analyze>
The description contains several hallucinations:
1. "bright red" - The car is actually blue
2. "sports car" - It's a regular sedan
3. "racing" - The car is parked
4. "lake" - There is no water body in the image
5. "dragons" - No mythical creatures present
</Analyze>
<pre_output>
The <hallucination>bright red</hallucination> <hallucination>sports</hallucination> car is <hallucination>quickly racing</hallucination> past a <hallucination>beautiful lake</hallucination> with <hallucination>three</hallucination> <hallucination>flying</hallucination> <hallucination>dragons</hallucination>.
</pre_output>"""


COMPRESSED_SYSTEM_MESSAGE = """You are a hallucination detector for multimodal large language models. Your task is to:
1. Analyze the image and the model's response to an image-related query.
2. In <Analyze>...</Analyze> tags:
   - List hallucinations: 
     - Identify each hallucinated detail
     - Explain why it's a hallucination
3. In <pre_output>...</pre_output> tags:
   - Output the <b>original model's response unchanged</b> with <hallucination> tags
   - Tag hallucinated words/phrases with <hallucination>
   - If no hallucinations, output the original text unchanged

Example Input:
prompt given to the model: describe the image
model's response: The bright red sports car...

Example Output Format:
<Analyze>
1. "bright red" - Car is blue
2. "sports car" - It's a sedan
</Analyze>
<pre_output>
The <hallucination>bright red</hallucination> <hallucination>sports</hallucination> car...
</pre_output>

"""

# 用户提词模板
USER_PROMPT_TEMPLATE = """Here is the prompt given to the model:
{prompt}

Here is the model's response:
{test_description}

Please analyze the image and add <hallucination> tags to any hallucinated content in the model's response. Remember to tag each hallucinated concept separately!"""

BASIC_USER_PROMPT_TEMPLATE = """Here is the prompt given to the model:
{prompt}

Here is the model's response:
{test_description}

Please analyze the image and add <hallucination> tags to any hallucinated content in the model's response. Remember to tag each hallucinated content separately!"""

# 数学题专用的系统提示词
MATH_SYSTEM_MESSAGE = """You are a precise hallucination detector specialized in mathematical reasoning and geometric problem solving. Your task is to identify and tag both direct and indirect hallucinations in mathematical solutions.

Hallucination Types:
1. Direct Hallucinations (Visual Misinterpretation):
   - Incorrect reading of numerical values from the image
   - Misinterpretation of geometric relationships
   - Wrong understanding of geometric conditions
   - Misidentification of geometric shapes or angles

2. Indirect Hallucinations (Consequential Errors):
   - Calculations using incorrectly read values
   - Deductions based on misinterpreted geometric relationships
   - Conclusions drawn from incorrect premises
   - Any subsequent errors caused by direct hallucinations

Your Task:
1. Carefully examine the geometric figure in the image
2. Read the question and the model's solution
3. First provide your analysis in <Analyze>...</Analyze> tags:
   - List each hallucinated detail (both direct and indirect)
   - Identify whether it's a direct or indirect hallucination
   - Explain the relationship between indirect and direct hallucinations

4. Then in <pre_output>...</pre_output> tags:
   - ONLY output the original solution with added <hallucination> tags
   - DO NOT add any additional text, analysis, or explanations
   - DO NOT modify the original text structure or wording
   - Tag EACH specific mathematical detail that contains hallucinations
   - Be as granular as possible in tagging:
     * Individual numbers and measurements
     * Mathematical relationships and equations
     * Geometric properties and conditions
     * Reasoning steps and conclusions
   - Include confidence scores (1-10):
     * 10: Absolute certainty of hallucination
     * 8-9: Clear visual evidence contradicts this
     * 6-7: Significant discrepancy with image
     * 4-5: Moderate uncertainty
     * 1-3: Slight suspicion of error

Example:
<Analyze>
The solution contains both direct and indirect hallucinations:

Direct Hallucinations:
1. "angle ABC = 60°" - Direct visual error, angle is clearly 45°
2. "length AB = 8" - Misread value, actually marked as 6

Indirect Hallucinations:
3. "cos(60°) = 0.5" - Based on wrong angle
4. "AB × cos(60°) = 4" - Uses both wrong angle and length
5. "final answer is 4" - Conclusion based on previous errors
</Analyze>

<pre_output>
In triangle ABC, <hallucination>angle ABC = 60°</hallucination> and <hallucination>length AB = 8</hallucination>. Using these values, <hallucination>cos(60°) = 0.5</hallucination>, so <hallucination>AB × cos(60°) = 4</hallucination>. Therefore, <hallucination>the answer is 4</hallucination>.
</pre_output>

Note: The <pre_output> section should ONLY contain the original solution with added hallucination tags. DO NOT include any additional text or explanations within the <pre_output> tags. Keep the content of the original solution text strictly unchanged!!!!"""

# 数学题专用的用户提示词模板
MATH_USER_PROMPT_TEMPLATE = """Please analyze this geometry problem and its solution:

Question:
{prompt}

Model's Solution:
{test_description}

Your task:
1. Carefully examine the geometric figure
2. Identify and tag ALL hallucinations in the solution:
   - Direct hallucinations from visual misinterpretation
   - Indirect hallucinations caused by earlier errors
3. Use <hallucination> tags with appropriate confidence scores
4. Be as specific and granular as possible in your tagging
5. Consider both numerical values and geometric relationships
6. If the solution is correct, output the original text unchanged

Remember to analyze the entire solution chain and tag both initial errors and their consequences!
"""

MATH_COMPRESSED_SYSTEM_MESSAGE = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to identify and tag hallucinations in solutions.

**Your Task:**
1. Examine the geometric figure and the model's solution.
2. Provide analysis in <Analyze>...</Analyze> tags:
   - List hallucinations and explain why they are hallucinations

3. In <pre_output>...</pre_output> tags:
   - Output the <b>original model's solution unchanged</b> with added <hallucination> tags
   - Tag each mathematical detail with hallucinations
   - Be specific: numbers, measurements, equations, reasoning, conclusions
   - Do not modify original text or structure

**Example Input:**
Question: What is the cosine value of angle ABC in triangle ABC?
Model's Solution: In triangle ABC, angle ABC = 60°. Using this, cos(60°) = 0.5.

**Example Output:**
<Analyze>
1. "angle ABC = 60°" - Actually 45°
2. "cos(60°) = 0.5" - Based on wrong angle
</Analyze>

<pre_output>
In triangle ABC, <hallucination>angle ABC = 60°</hallucination>. Using this, <hallucination>cos(60°) = 0.5</hallucination>.
</pre_output>
"""

# vanilla系统提示词
BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to tag hallucinations in mathematical solutions.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
1. Start with EXACTLY this line: "Here is the response with hallucinated content tagged:"
2. Then use <Tagged_Text> tags to wrap the tagged response
3. Inside <Tagged_Text> tags:
   - Output the original solution with ONLY <hallucination> tags added
   - DO NOT modify or change any words in the original solution
   - ONLY add <hallucination> tags around hallucinated content
   - If there are no hallucinations, output the original text exactly as is
4. End with </Tagged_Text>
5. DO NOT add any other text, analysis, or explanation
6. ANY OTHER FORMAT WILL BE REJECTED

Example Input:
Question: What is the cosine value of angle ABC in triangle ABC?
Model's Solution: In triangle ABC, angle ABC = 60°. Using this, cos(60°) = 0.5.

Correct Output Format:
Here is the response with hallucinated content tagged:
<Tagged_Text>
In triangle ABC, <hallucination>angle ABC = 60°</hallucination>. Using this, <hallucination>cos(60°) = 0.5</hallucination>.
</Tagged_Text>

INCORRECT Outputs (DO NOT DO THESE):
❌ Any text before "Here is the response with hallucinated content tagged:"
❌ Any text between the header and <Tagged_Text>
❌ Any text after </Tagged_Text>
❌ Any explanatory text or analysis
❌ Any modification to the original text
❌ Any additional formatting or tags besides <hallucination>
"""

BASIC_COMPRESSED_SYSTEM_MESSAGE = """You are a hallucination detector for multimodal large language models. Your task is to tag hallucinations in the model's response.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
1. Start with EXACTLY this line: "Here is the response with hallucinated content tagged:"
2. Then use <Tagged_Text> tags to wrap the tagged response
3. Inside <Tagged_Text> tags:
   - Output the original response with ONLY <hallucination> tags added
   - DO NOT modify or change any words in the original response
   - ONLY add <hallucination> tags around hallucinated content
   - If there are no hallucinations, output the original text exactly as is
4. End with </Tagged_Text>
5. DO NOT add any other text, analysis, or explanation
6. ANY OTHER FORMAT WILL BE REJECTED

Example Input:
prompt given to the model: describe the image
model's response: The bright red sports car is parked near a lake.

Correct Output Format:
Here is the response with hallucinated content tagged:
<Tagged_Text>
The <hallucination>bright red</hallucination> <hallucination>sports</hallucination> car is <hallucination>parked near a lake</hallucination>.
</Tagged_Text>

INCORRECT Outputs (DO NOT DO THESE):
❌ Any text before "Here is the response with hallucinated content tagged:"
❌ Any text between the header and <Tagged_Text>
❌ Any text after </Tagged_Text>
❌ Any explanatory text or analysis
❌ Any modification to the original text
❌ Any additional formatting or tags besides <hallucination>
"""

# 2-shot系统提示词
BASIC_COMPRESSED_SYSTEM_MESSAGE_2_SHOT = """You are a hallucination detector for multimodal large language models. Your task is to tag hallucinations in the model's response.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
1. Start with EXACTLY this line: "Here is the response with hallucinated content tagged:"
2. Then use <Tagged_Text> tags to wrap the tagged response
3. Inside <Tagged_Text> tags:
   - Output the original response with ONLY <hallucination> tags added
   - DO NOT modify or change any words in the original response
   - ONLY add <hallucination> tags around hallucinated content
   - If there are no hallucinations, output the original text exactly as is
4. End with </Tagged_Text>
5. DO NOT add any other text, analysis, or explanation
6. ANY OTHER FORMAT WILL BE REJECTED

Example Input 1:
prompt given to the model: describe the image
model's response: The bright red sports car is parked near a lake.

Example Output 1:
Here is the response with hallucinated content tagged:
<Tagged_Text>
The <hallucination>bright red</hallucination> <hallucination>sports</hallucination> car is <hallucination>parked near a lake</hallucination>.
</Tagged_Text>

Example Input 2:
prompt given to the model: what is the person wearing?
model's response: The woman is wearing a blue dress with white flowers and holding a black umbrella.

Example Output 2:
Here is the response with hallucinated content tagged:
<Tagged_Text>
The <hallucination>woman</hallucination> is wearing a <hallucination>blue dress with white flowers</hallucination> and <hallucination>holding a black umbrella</hallucination>.
</Tagged_Text>

INCORRECT Outputs (DO NOT DO THESE):
❌ Any text before "Here is the response with hallucinated content tagged:"
❌ Any text between the header and <Tagged_Text>
❌ Any text after </Tagged_Text>
❌ Any explanatory text or analysis
❌ Any modification to the original text
❌ Any additional formatting or tags besides <hallucination>
"""

BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE_2_SHOT = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to tag hallucinations in mathematical solutions.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
1. Start with EXACTLY this line: "Here is the response with hallucinated content tagged:"
2. Then use <Tagged_Text> tags to wrap the tagged response
3. Inside <Tagged_Text> tags:
   - Output the original solution with ONLY <hallucination> tags added
   - DO NOT modify or change any words in the original solution
   - ONLY add <hallucination> tags around hallucinated content
   - If there are no hallucinations, output the original text exactly as is
4. End with </Tagged_Text>
5. DO NOT add any other text, analysis, or explanation
6. ANY OTHER FORMAT WILL BE REJECTED

Example Input 1:
Question: What is the cosine value of angle ABC in triangle ABC?
Model's Solution: In triangle ABC, angle ABC = 60°. Using this, cos(60°) = 0.5.

Example Output 1:
Here is the response with hallucinated content tagged:
<Tagged_Text>
In triangle ABC, <hallucination>angle ABC = 60°</hallucination>. Using this, <hallucination>cos(60°) = 0.5</hallucination>.
</Tagged_Text>

Example Input 2:
Question: Find the area of the rectangle ABCD.
Model's Solution: The length of rectangle ABCD is 8 cm and width is 5 cm. Therefore, the area is 8 × 5 = 40 square cm.

Example Output 2:
Here is the response with hallucinated content tagged:
<Tagged_Text>
The <hallucination>length of rectangle ABCD is 8 cm</hallucination> and <hallucination>width is 5 cm</hallucination>. Therefore, <hallucination>the area is 8 × 5 = 40 square cm</hallucination>.
</Tagged_Text>

INCORRECT Outputs (DO NOT DO THESE):
❌ Any text before "Here is the response with hallucinated content tagged:"
❌ Any text between the header and <Tagged_Text>
❌ Any text after </Tagged_Text>
❌ Any explanatory text or analysis
❌ Any modification to the original text
❌ Any additional formatting or tags besides <hallucination>
"""

# 幻觉类型系统提示词
HALLUCINATION_TYPE_SYSTEM_MESSAGE = """You are a hallucination detector for multimodal large language models. Your task is to tag hallucinations in the model's response.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
1. Start with EXACTLY this line: "Here is the response with hallucinated content tagged:"
2. Then use <Tagged_Text> tags to wrap the tagged response
3. Inside <Tagged_Text> tags:
   - Output the original response with ONLY <hallucination> tags added
   - DO NOT modify or change any words in the original response
   - ONLY add <hallucination> tags around hallucinated content
   - If there are no hallucinations, output the original text exactly as is
4. End with </Tagged_Text>
5. DO NOT add any other text, analysis, or explanation
6. ANY OTHER FORMAT WILL BE REJECTED

When identifying hallucinations, refer to these types:
- Object: Misidentify objects in the image
- OCR: Misread text or numbers in the image
- Numerical Attribute: Misread quantities, sizes, measurements
- Color Attribute: Misidentify colors of objects
- Shape Attribute: Misinterpret shapes of objects
- Spatial Attribute: Misread positions, orientations, distances
- Numerical Relations: Misinterpret quantitative comparisons
- Size Relations: Misread relative sizes between objects
- Spatial Relations: Misinterpret positions between objects
- Logical Errors: Make mistakes in reasoning steps
- Query Misunderstanding: Misunderstand the query intent

Example Input:
prompt given to the model: describe the image
model's response: The bright red sports car is parked near a lake.

Correct Output Format:
Here is the response with hallucinated content tagged:
<Tagged_Text>
The <hallucination>bright red</hallucination> <hallucination>sports</hallucination> car is <hallucination>parked near a lake</hallucination>.
</Tagged_Text>

INCORRECT Outputs (DO NOT DO THESE):
❌ Any text before "Here is the response with hallucinated content tagged:"
❌ Any text between the header and <Tagged_Text>
❌ Any text after </Tagged_Text>
❌ Any explanatory text or analysis
❌ Any modification to the original text
❌ Any additional formatting or tags besides <hallucination>
"""

MATH_HALLUCINATION_TYPE_SYSTEM_MESSAGE = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to tag hallucinations in mathematical solutions.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
1. Start with EXACTLY this line: "Here is the response with hallucinated content tagged:"
2. Then use <Tagged_Text> tags to wrap the tagged response
3. Inside <Tagged_Text> tags:
   - Output the original solution with ONLY <hallucination> tags added
   - DO NOT modify or change any words in the original solution
   - ONLY add <hallucination> tags around hallucinated content
   - If there are no hallucinations, output the original text exactly as is
4. End with </Tagged_Text>
5. DO NOT add any other text, analysis, or explanation
6. ANY OTHER FORMAT WILL BE REJECTED

When identifying hallucinations, refer to these types:
- OCR: Misread numbers, variables, or mathematical symbols
- Numerical Attribute: Misread measurements, angles, lengths
- Shape Attribute: Misinterpret geometric shapes
- Spatial Attribute: Misread positions, orientations
- Numerical Relations: Misinterpret quantitative comparisons
- Size Relations: Misread relative sizes
- Spatial Relations: Misinterpret geometric relationships
- Logical Errors: Make mistakes in reasoning steps
- Calculation Errors: Perform incorrect mathematical operations
- Knowledge Errors: Apply incorrect formulas or concepts
- Query Misunderstanding: Misunderstand the problem

Example Input:
Question: What is the cosine value of angle ABC in triangle ABC?
Model's Solution: In triangle ABC, angle ABC = 60°. Using this, cos(60°) = 0.5.

Correct Output Format:
Here is the response with hallucinated content tagged:
<Tagged_Text>
In triangle ABC, <hallucination>angle ABC = 60°</hallucination>. Using this, <hallucination>cos(60°) = 0.5</hallucination>.
</Tagged_Text>

INCORRECT Outputs (DO NOT DO THESE):
❌ Any text before "Here is the response with hallucinated content tagged:"
❌ Any text between the header and <Tagged_Text>
❌ Any text after </Tagged_Text>
❌ Any explanatory text or analysis
❌ Any modification to the original text
❌ Any additional formatting or tags besides <hallucination>
"""

# 推理和标记系统提示词
REASON_AND_TAG_SYSTEM_MESSAGE = """You are a hallucination detector for multimodal large language models. Your task is to:
1. Analyze the image and the model's response to an image-related query.
2. First provide your analysis in <Analysis>...</Analysis> tags:
   - Analyze what is actually present in the image
   - Compare it with what the model claims
   - Explain any discrepancies you find
3. Then in <Tagged_Text>...</Tagged_Text> tags:
   - Output the original model's response unchanged with <hallucination> tags
   - Tag hallucinated words/phrases with <hallucination>
   - If no hallucinations, output the original text unchanged

Example Input:
prompt given to the model: describe the image
model's response: The bright red sports car...

Example Output Format:
<Analysis>
The image shows a car, but:
1. The car is actually blue, not red
2. It's a regular sedan, not a sports car
Therefore, both the color description and car type are hallucinations.
</Analysis>

<Tagged_Text>
The <hallucination>bright red</hallucination> <hallucination>sports</hallucination> car...
</Tagged_Text>
"""

MATH_REASON_AND_TAG_SYSTEM_MESSAGE = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to identify and tag hallucinations in solutions.

**Your Task:**
1. Examine the geometric figure and the model's solution.
2. First provide your analysis in <Analysis>...</Analysis> tags:
   - Analyze the actual geometric properties in the figure
   - Compare them with the model's claims
   - Explain any mathematical discrepancies
   - Point out any incorrect calculations or logical errors
3. Then in <Tagged_Text>...</Tagged_Text> tags:
   - Output the original model's solution unchanged with <hallucination> tags
   - Tag each mathematical detail with hallucinations
   - Do not modify original text or structure

**Example Input:**
Question: What is the cosine value of angle ABC in triangle ABC?
Model's Solution: In triangle ABC, angle ABC = 60°. Using this, cos(60°) = 0.5.

**Example Output:**
<Analysis>
Examining the geometric figure:
1. The angle ABC is actually 45° based on the markings, not 60°
2. Due to this error, the cosine calculation is also wrong
3. The correct value should be cos(45°) = √2/2 ≈ 0.707
Therefore, both the angle measurement and its cosine value are hallucinations.
</Analysis>

<Tagged_Text>
In triangle ABC, <hallucination>angle ABC = 60°</hallucination>. Using this, <hallucination>cos(60°) = 0.5</hallucination>.
</Tagged_Text>
"""

# 数据集配置
DATASET_CONFIGS = {
    # "multimath_300k": DatasetConfig(
    #     name="multimath_300k",
    #     data_path="data/processed_multimath_300k.json",
    #     output_path="results/evaluation_results_multimath_300k.json",
    #     system_message=MULTIMATH_SYSTEM_MESSAGE,
    #     user_prompt_template=MULTIMATH_USER_PROMPT_TEMPLATE
    # ),
    
    "rlhfv": DatasetConfig(
        name="rlhfv",
        data_path="data/processed_rlhfv_dataset.json",
        output_path="results/tmp/evaluation_results_rlhfv.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "mhal": DatasetConfig(
        name="mhal",
        data_path="data/processed_mhal_dataset.json",
        output_path="results/tmp/evaluation_results_mhal.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "geo_170k": DatasetConfig(
        name="geo_170k",
        data_path="data/processed_geo_170k.json",
        output_path="results/tmp/evaluation_results_geo_170k.json",
        system_message=BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "mathv_360k": DatasetConfig(
        name="mathv_360k",
        data_path="data/processed_mathv_360k.json",
        output_path="results/tmp/evaluation_results_mathv_360k.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "test": DatasetConfig(
        name="test",
        data_path="data/processed_test.json",
        output_path="results/tmp/evaluation_results_test.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    )
} 


