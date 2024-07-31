from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)

# Simple Text Generation
# pipe=pipeline(
#     task="text-generation",
#     model=model,
#     do_sample=True,
#     tokenizer=tokenizer,
#     max_new_tokens=30,
#     temperature=0.7
# )

# local_llm = HuggingFacePipeline(pipeline=pipe)

# chain = local_llm | StrOutputParser()
# print(chain.invoke("The sea lion is a"))

# Output(PASS):
"""The sea lion is a marine mammal that is found in the Pacific Ocean.
It is a large animal that is about 10 feet long and weighs about 1,0"""

# Summarization: Seems to be unable to support summarization via pipeline.
# pipe=pipeline(
#     task="summarization",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=30
# )

# local_llm = HuggingFacePipeline(pipeline=pipe)

# print(local_llm("Sea lions are pinnipeds characterized by external ear flaps,
# long foreflippers, the ability to walk on all fours, short and thick hair,
# and a big chest and belly. Together with the fur seals, they make up the family Otariidae,
# eared seals. The sea lions have six extant and one extinct species (the Japanese sea lion)
# in five genera. Their range extends from the subarctic to tropical waters of the global
# ocean in both the Northern and Southern Hemispheres, with the notable exception of the
# northern Atlantic Ocean."))

# Output(FAIL):
"""Sea lions are pinnipeds characterized by external ear flaps, long foreflippers,
the ability to walk on all fours, short and thick hair, and a big chest and belly.
Together with the fur seals, they make up the family Otariidae, eared seals. The sea lions
have six extant and one extinct species (the Japanese sea lion) in five genera.
Their range extends from the subarctic to tropical waters of the global ocean in both
the Northern and Southern Hemispheres, with the notable exception of the northern Atlantic Ocean.
The sea lions are the largest pinnipeds in the world, with a body mass of 1,000–1,50"""

# For Questioning
# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# chain = prompt | local_llm | StrOutputParser()

# print("######## Print chain.invoke\n")
# print(chain.invoke("What is a sea lion?"))

# Output(PASS):
"""Question: What is a sea lion?

Answer: Let's think step by step.

A sea lion is a mammal that lives in the ocean.

A mammal is a type of animal that has a backbone."""

# For Summarization in LangChain
# template = """Summarize the following: {text}

# Answer: """

# prompt = PromptTemplate(template=template, input_variables=["text"])

# chain = prompt | local_llm | StrOutputParser()

# print("######## Print chain.invoke\n")
# print(chain.invoke(
#     """
#     'Sea lions are pinnipeds characterized by external ear flaps, long foreflippers,
# the ability to walk on all fours, short and thick hair, and a big chest and belly.
# Together with the fur seals, they make up the family Otariidae, eared seals. The sea lions
# have six extant and one extinct species (the Japanese sea lion) in five genera. Their range
# extends from the subarctic to tropical waters of the global ocean in both
# the Northern and Southern Hemispheres, with the notable exception of the northern Atlantic Ocean.'
#     """
# ))

# Output:
"""



*

*The sea lions are pinnipeds characterized by external ear flaps, long foreflippers, the ability to walk on
"""

# For Translation in LangChain
pipe = pipeline(task="translation", model=model, tokenizer=tokenizer, max_new_tokens=40, max_length=400)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """{text}

In {language}, this translates to: """

prompt = PromptTemplate(template=template, input_variables=["text", "language"])

chain = prompt | local_llm | StrOutputParser()

print("######## Print chain.invoke\n")
print(
    chain.invoke(
        {
            "text": """
        'Seekor rubah merah yang lincah melompati seekor anjing coklat yang malas.'
        """,
            "language": "English",
        }
    )
)

# Output(PASS):
"""
'Pada suatu masa dahulu, terdapat satu keluarga yang miskin dan ayahnya bekerja sebagai buruh di ladang getah.
Dalam hidup ini, Allah telah memberikan kita semua peluang untuk berjaya.'


In English, this translates to:

'In the past, there was a family that was poor and the father worked as a labourer in the rubber plantation.
In life, Allah
"""
