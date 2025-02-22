from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str  # Dataset name
    data_path: str  # Data file path
    output_path: str  # Output file path
    system_message: str  # System prompt
    user_prompt_template: str  # User prompt template
    tmp_dir: str = ""  # Temporary directory path


BASIC_USER_PROMPT_TEMPLATE = """Here is the prompt given to the model:
{prompt}

Here is the model's response:
{test_description}

Please analyze the image and add <hallucination> tags to any hallucinated content in the model's response. Remember to tag each hallucinated content separately!"""

# Vanilla system prompt
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

# 2-shot system prompt
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
Criteria_SYSTEM_MESSAGE = """You are a hallucination detector for multimodal large language models. Your task is to tag hallucinations in the model's response.

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

MATH_Criteria_SYSTEM_MESSAGE = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to tag hallucinations in mathematical solutions.

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


Analyze_then_judge_SYSTEM_MESSAGE = """You are a hallucination detector for multimodal large language models. Your task is to:
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

MATH_Analyze_then_judge_SYSTEM_MESSAGE = """You are a hallucination detector specializing in mathematical reasoning and geometry. Your task is to identify and tag hallucinations in solutions.

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

# Dataset configuration
DATASET_CONFIGS = {

    "RLHF-V": DatasetConfig(
        name="rlhfv",
        data_path="data/processed_rlhfv_dataset.json",
        output_path="results/tmp/evaluation_results_rlhfv.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "M-HalDetect": DatasetConfig(
        name="mhal",
        data_path="data/processed_mhal_dataset.json",
        output_path="results/tmp/evaluation_results_mhal.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "Geo170K": DatasetConfig(
        name="geo_170k",
        data_path="data/processed_geo_170k.json",
        output_path="results/tmp/evaluation_results_geo_170k.json",
        system_message=BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "MathV360K": DatasetConfig(
        name="mathv_360k",
        data_path="data/processed_mathv_360k.json",
        output_path="results/tmp/evaluation_results_mathv_360k.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    ),
    "MC": DatasetConfig(
        name="test",
        data_path="data/processed_test.json",
        output_path="results/tmp/evaluation_results_test.json",
        system_message=BASIC_COMPRESSED_SYSTEM_MESSAGE,
        user_prompt_template=BASIC_USER_PROMPT_TEMPLATE
    )
} 


