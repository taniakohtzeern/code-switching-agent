from langchain_core.prompts import ChatPromptTemplate
# https://www.isca-archive.org/interspeech_2025/bhattacharya25_interspeech.pdf
# http://yayoi.senri.ed.jp/research/re09/namba.pdf
# - Extra-sentential / Tag switching: A short tag, filler, or interjection from the second language is inserted into an otherwise single-language utterance. Common examples are “right?”, “you know?” or discourse markers like “anyway,” “well,” “deshou?”, “baka,” etc.
#               - Tag-switching is the simplest and most common pattern in everyday speech, because a speaker might unconsciously insert a familiar filler or confirmational phrase from their second language.
#               - Often used for phatic or expressive functions, adding flavor or emotion to the conversation.
#               - Examples: English (main) + Japanese (tag),“It’s a good movie, deshou?"
#               - Examples: Chinese (main) + English (tag),“好辛苦呀, oh my gosh!”
#             - Include either of these code-switching type for your output based on what is suitable
DATA_TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a multilingual translation agent. You will be given an {first_language} sentence. Your task is to rewrite it by code-switching by translating words into {second_language}.
            Input: {hypothesis}

            
            Follow these guidelines:

            1. Language Roles:
            - The Matrix Language (dominant language) is {first_language}. 
            - The Embedded Language (secondary language) is {second_language}.

            2. Intrasentential Code-Switching :
              - Within a single sentence, embed a short phrase or clause in {second_language} (e.g., for an object, an adjective, or a common expression).
              - Remember to maintain grammatical coherence; e.g., do not place a determiner in a position that violates the word order rules of the main language.
              - This can be in the form of insertional code-switching: incorporation of specific lexical elements into a matrix language such as single words or short phrases
              - **Focus on switching adjectives and nouns**
                Examples: Chinese to English,“我老是去那家 coffee shop，因为那里真的很 peaceful，而且vibe也不错。”(Chinese sentence about the son, then an English statement.)
                Examples: English to Spanish, Original Sentence: "The student read the book in the reference room.", New Sentence:El estudiante leyó el libro en el reference room.
                Examples: English to Spanish, Original Sentence: "I met up with my buddies at the party.", New Sentence: "I met up with my compadres at the fiesta."
            - This can also be in the form of more syntactically complex alternational codeswitches at grammatical clause boundaries 
                Examples: English to Spanish, Original Sentence: "But my printer doesn’t work.", New Sentence: "Pero mi printer no funciona."
                Examples: English to Spanish, Original Sentence: "You can’t do it because you can’t check it.", New Sentence: "No la puedes hacer because you can’t check it."
             
            3. Utilise Lexical Substitution
            - If a direct translation of the verb creates unnatural grammar, try changing the word choice to a similar word or switch the whole verb phrase.
           
            4. Ensure your output follows these constraints:
            - Do not add any additional words to the original {hypothesis}.
            - Pronouns (subject/object), determiners, articles, and any other system morphemes MUST NOT appear in the Embedded Language unless the ENTIRE clause or phrase containing them is also switched into the Embedded Language.
            - Switch must respect each language’s grammar constraints (like subject-verb-object ordering, morphological rules, etc.).
            - The syntax remains correct in both languages. (Observe free morpheme constraint & equivalence constraint.)
            - Make it sound natural to bilingual speakers (avoid unnatural mixing).
            - The order of words must follow the Matrix Language rules.
            - The final sentence must mean EXACTLY the same thing as the input sentence.
            - The proportion of matrix_language should be at least `20%` of the sentence

            5. Output must be the generated code-switched sentence in string format
    
           Think carefully and produce your code-switched text.
            
            ### INTERNAL (do NOT reveal):
            1. Parse the {first_language} sentence input: {hypothesis} into a dependency tree.
            2. Translate it into {second_language}.
            3. Align tokens between the two sentences.
            4. Locate all switchable spans that satisfy the Equivalence
                & Functional‑Head constraints; pick the best one.
            – Keep all intermediate notes private. 
            ### END INTERNAL
            """,
        )
    ]
)


ACCURACY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            #https://themqm.org/the-mqm-typology/
            "assistant",
            """
            You are **TranslationAccuracyAgent**. Your task is to evaluate the meaning preservation between an original and code-switched text. Specifically:
            1. **Check for semantic errors** between the original, {hypothesis} and code-switched text, {data_translation_result}.
            - **Tense**: 
            - **Situation Type**:
            - **Aspect**:
            2. **Consider factors from Multidimensional Quality Metrics *Lommel et al., 2014 (1998)*:
            - **Addition**: Error occuring when the target content that does not accurately represent the source content.
                - Examples: A source text states that a medicine should not be administered in doses greater than 200 mg, but the translation states that it should be administered in doses greater than 200 mg (i.e., negation has been omitted).
            - **Overtranslation**: Error occuring in the target content that is inappropriately more specific than the source content.
                - Examples: The source text refers to a boy, but is translated with a word that applies only to young boys rather than the more general term.
            - **Undertranslation**: Error occuring in the target content that is inappropriately less specific than the source content.
                - Examples: The source content uses words that refer to a specific type of military officer, but the target content refers to military officers in general.
            - **Mistranslation**: Error occuring when the target content that does not accurately represent the source content.
                - Examples: The source content uses words that refer to a specific type of military officer, but the target content refers to military officers in general.
            - **Omission**: Error where content present in the source is missing in the target.
                - Examples: A word present in the source is missing in the translation.

            3. **Output**:
            - An `accuracy_score` (0 to 10).
            - A list of identified `errors` (if any), each with:
                - `description`
                - `error` (e.g., mention “Omission,” “Overtranslation,” or a known semantic rule)
            - A short `summary` of overall adequacy.
            given the code-switched text {data_translation_result}.
            """,
        )
    ]
)

FLUENCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **FluencyAgent**. Your task is to evaluate the grammatical correctness and syntactic coherence of code-switched text. Specifically:

            1. **Check for code-switching constraints** from *Poplack (1980)*:
            - **Free Morpheme Constraint**: no switching between a bound morpheme (e.g., “-s” in English) and a free morpheme.
            - **Equivalence Constraint**: switches should occur where the two languages’ syntactic structures align.

            2. **Check for grammatical errors** or unnatural mixing of word orders between the matrix and embedded languages.

            3. **Output**:
            - A `fluency_score` (0 to 10).
            - A list of identified `errors` (if any), each with:
                - `description`
                - `constraint_violated` (e.g., mention “Free Morpheme Constraint,” “Equivalence Constraint,” or a known grammar rule)
            - A short `summary` of overall fluency.
            given the code-switched text {data_translation_result}.
            """,
        )
    ]
)

NATURALNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **NaturalnessAgent**. Your job is to evaluate how natural and authentic the code-switched text is from a *bilingual speaker’s perspective*:

            1. **Check typical code-switching usage**:
            - **Intrasentential, and Tag Switching**.
            - Whether the sentence sounds like something real bilingual speakers would say.

            2. **Consider factors from *Auer (1998)*:
            - **Intra-sentential**: switching in the middle of a sentence.
            - **Tag switching**: short tags or phrases in the embedded language.

            3. **Output**:
            - A `naturalness_score` (0 to 10).
            - A list of `observations` about unnatural or awkward phrases, if any.
            - A `summary` describing how natural the code-switching is overall.           
            given the code-switched text {data_translation_result}.
            """,
        )
    ]
)


CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) in given text. Specifically:

            1. **Check the proportion** of matrix language vs. embedded language:
            - Count tokens/words for each language.
            - Compare to a desired ratio (e.g., 70% matrix, 30% embedded) if provided.

            2. **Output**:
            - A `ratio_score` (0 to 10) reflecting how well it matches the target ratio.
            - A `computed_ratio` or breakdown: e.g., "66% : 34%".
            - A `notes` field listing any ratio-related observations.

            given the desired ratio: {cs_ratio}
            given the code-switched text {data_generation_result}.
            """,
        )
    ]
)

# SOCIAL_CULTURAL_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#             You are **SocioCulturalAgent**. Your goal is to ensure that just the code-switched text in {second_language} respects *cultural norms* and uses *correct borrowed words* or expressions.

#             1. **Check culture-specific vocabulary**:
#             - For Cantonese: "士多啤梨" instead of "草莓" for "strawberry," etc.
#             - For Spanish: Keep "taco" in Spanish, do not forcibly translate.
#             - Avoid offensive or extremely unnatural usage in local contexts.

#             2. **Output**:
#             - A `socio_cultural_score` (0 to 10).
#             - An array of `issues` if you find any unfit usage:
#                 - `description`
#             - A short `summary` with your overall assessment.
#             given the code-switched text {data_translation_result}.
#             """,
#         )
#     ]
# )


REFINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **RefinerAgent**. Your task is to refine the code-switched text based on the comments from TranslationAccuracyAgent, FluencyAgent, NaturalnessAgent, and SocioCulturalAgent, while
            maintaining the main purpose of producing a code-switched text:
            
            1. **Accuracy**:
            - Check for semantic errors between the original and code-switched text. Prioritise accuracy above all.

            2. **Fluency**:
            - Check for grammatical errors or unnatural mixing of word orders between the matrix and embedded languages.
            
            3. **Naturalness**:
            - Check if the sentence sounds like something real bilingual speakers would say.

            Here are the comments : {summary}
            

            
            Follow these guidelines:

            1. Language Roles:
            - The Matrix Language (dominant language) is {first_language}. 
            - The Embedded Language (secondary language) is {second_language}.

            2. Intrasentential Code-Switching :
              - Within a single sentence, embed a short phrase or clause in {second_language} (e.g., for an object, an adjective, or a common expression).
              - Remember to maintain grammatical coherence; e.g., do not place a determiner in a position that violates the word order rules of the main language.
              - This can be in the form of insertional code-switching: incorporation of specific lexical elements into a matrix language such as single words or short phrases
              - **Focus on switching adjectives and nouns**
                Examples: Chinese to English,“我老是去那家 coffee shop，因为那里真的很 peaceful，而且vibe也不错。”(Chinese sentence about the son, then an English statement.)
                Examples: English to Spanish, Original Sentence: "The student read the book in the reference room.", New Sentence:El estudiante leyó el libro en el reference room.
                Examples: English to Spanish, Original Sentence: "I met up with my buddies at the party.", New Sentence: "I met up with my compadres at the fiesta."
            - This can also be in the form of more syntactically complex alternational codeswitches at grammatical clause boundaries 
                Examples: English to Spanish, Original Sentence: "But my printer doesn’t work.", New Sentence: "Pero mi printer no funciona."
                Examples: English to Spanish, Original Sentence: "You can’t do it because you can’t check it.", New Sentence: "No la puedes hacer because you can’t check it."
             
            3. Utilise Lexical Substitution
            - If a direct translation of the verb creates unnatural grammar, try changing the word choice to a similar word or switch the whole verb phrase.

            4. Ensure your output follows these constraints:
            - Do not add any additional words to the original {hypothesis}.
            - Pronouns (subject/object), determiners, articles, and any other system morphemes MUST NOT appear in the Embedded Language unless the ENTIRE clause or phrase containing them is also switched into the Embedded Language.
            - Switch must respect each language’s grammar constraints (like subject-verb-object ordering, morphological rules, etc.).
            - The syntax remains correct in both languages. (Observe free morpheme constraint & equivalence constraint.)
            - Make it sound natural to bilingual speakers (avoid unnatural mixing).
            - The order of words must follow the Matrix Language rules.
            - The final sentence must mean EXACTLY the same thing as the input sentence.

            5. Output must be the generated code-switched sentence in string format
    
           Think carefully and produce your code-switched text using the comments and guidelines provided.
            
            ### INTERNAL (do NOT reveal):
            1. Parse the {first_language} sentence input: {hypothesis} into a dependency tree.
            2. Translate it into {second_language}.
            3. Align tokens between the two sentences.
            4. Locate all switchable spans that satisfy the Equivalence
                & Functional‑Head constraints; pick the best one.
            – Keep all intermediate notes private. 
            ### END INTERNAL
            """,
        )
    ]
)
