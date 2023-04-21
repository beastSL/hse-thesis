# Annotation instructions

This document describes the annotation instructions for subjectivity detection in news articles. First, we will describe what should be treated as subjectivity and objectivity and how to spot it. Then, we will fully describe the annotation task.

## Subjectivity and objectivity
First, we would like to note there are no formal definitions of subjectivity and objectivity. In a lot of cases, you will have to appeal to your intuition and the reaction after reading a sentence. However, we will try to give advice on how to notice subjectivity and help build up the needed intuition.

### Types of subjectivity
Subjectivity is an expression that represents opinions, beliefs, thoughts, feelings, emotions, goals, evaluations, and/or judgments (we'll use the expression "private states" to generalize these concepts). One key sign of it is that it is not open to objective observation or verification. Subjective expressions are the ones that cannot be refuted or confirmed.

The three main types of subjectivity are:
- direct mentions of private state. This is the easiest-to-spot sign of subjectivity, and it includes words that directly indicate private states. Examples:
  - The U.S. fears a spill-over.
  - I really enjoyed the book I read last night.

  These sentences are highly subjective because the words "fear" directly indicates the emotion of fear, and the word "enjoyed" directly indicates the emotion of joy.
- expressive subjective elements. These are words or expressions in the text that implicitly indicate subjectivity. Here are some examples:
  - The report is full of absurdities.
  - The sunset painted the sky with a fiery orange hue.

  We think these sentences are subjective, because the expressions "full of absurdities" and "fiery orange hue" represent the writers opinions or emotions.
- subjective speech events. When labeling the sentences which contain direct or indirect speech, you should not base your judgment on the contents of the speech. Instead, you should base your judgments on the words the point to the speech, such as "said" or "called", as well as the other parts of the sentence. Examples:
  - Sargeant O’Leary said the incident took place at 2:00pm.
  - Defence officials accused Beijing of using President Tsai's US visit as an "excuse to conduct military exercises".

  Here, we think the first sentence is objective, because the word "said" does not indicate any private states, and the second sentence in subjective, because the word "accused" does indicate accusation, which is a belief.

Note that this is not an exhaustive list of subjectivity types. In some cases, you'll need to apply your intuition and appeal to the reaction a sentence gives you.

The sentences that do not contain any subjectivity above and that present statements are considered objective. Note that these statements are not necessarily correct. Some examples:
- The Earth is flat.
- The Dow Jones Industrial Average closed at 34,035.99 points on Monday.

### Other important advice
The other important things we need to mention before describing the exact task:
- There are no fixed rules about how particular words should be annotated. The instructions describe the annotations of specific examples, but do not state that specific words should always be annotated a certain way.
- Sentences should be interpreted with respect to the contexts in which they appear. The annotators should not take sentences out of context and think
what they could mean, but rather should judge them as they are being used in that particular sentence and document.
- It is impossible to cover all types of sentences in this instruction. For example, there could be sentences containing both objective and subjective elements. The subjective elements can also play a minor role in the sentence. You should base your judgment on your inner reaction and intuition after reading a sentence.

## Task
You will be consequently given sentences from a newspaper. Every sentence will be surrounded by several adjacent sentences to provide context, but the current sentence you're labeling will be highlighted.

Please note that some sentences might contain explicit language, since the papers were scraped from the Internet.

Your task is to assign each sentence a subjectivity score. The score will be measured on a discrete scale from 1 to 5. 1 corresponds to a completely objective sentence, 5 -- to a completely subjective one. You will also be given an option to assign an "Not Applicable" label. Here are the explanations of the scale (also, see below for examples):
- "Not Applicable" label is used when the instruction does not cover the sentence. Note that since scraping the data contains some amount of noise, you should put the "Not Applicable" label if the sentence fully consists of noise. Other cases for the "Not Applicable" label are incomplete sentences, questions and sentences that don't contain any statements.
- Scores 1 and 5 should be assigned when you are confident that the sentence is objective or subjective.
- Scores 2 and 4 should be assigned when you are unconfident but suspect that the sentence is objective or subjective.
- Score 3 should be assigned if a sentence presents a statement but it is difficult to say whether the sentence is objective or subjective.

In the end, we'd like to provide some examples of the sentences for each score.
- Not Applicable: 
  - What is happening across the UK?
  - Monday, April 10.
  - Find out more!
  - Snapchat Twitter Instagram LinkedIn Facebook
  - Am I a worthless creature (Heh.
  - ) or do I have the right?
- Score 1
  - Arctic Monkeys released a new album yesterday.
  - Nurses have rejected the government's pay offer and will now go on strike.
- Score 2
  - Arctic Monkeys released a highly anticipated new album yesterday.
  - Freya Ridings and Alexis Ffrench will grace the stage in the grounds of Windsor Castle, the BBC has confirmed.
- Score 3
  - Arctic Monkeys can release a highly anticipated new album soon.
  - The stock market can be unpredictable and volatile.
- Score 4
  - My favourite band Arctic Monkeys can release a highly anticipated new album soon.
  - The performance of the athlete was impressive, especially considering the challenging weather conditions.
- Score 5
  - I was amazed by the new album of Arctic Monkeys.
  - I strongly believe that everyone should have access to quality healthcare, and it's a fundamental human right.