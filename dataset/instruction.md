# Annotation instructions

This document describes the annotation instructions for subjectivity detection in news articles. First, we will describe what should be treated as subjectivity and objectivity and how to spot it. Then, we will fully describe the annotation task.

## Subjectivity and objectivity
First, we would like to note there are no formal definitions of subjectivity and objectivity. However, we will try to give advice on how to notice a subjectivity and words that can mark it.

Private state can be described as a general term that covers opinions, beliefs, thoughts, feelings, emotions, goals, evaluations, and judgments. It is a state that is not open to objective observation or verification. As an example, a person may be observed to assert that God exists, but not to believe God exists. Here, belief expresses the private states. 

We will call private state expression subjectivity. The three main types of subjectivity are:
- explicit mentions of private state;
- expressive subjective elements;
- speech events expressing private states.

Explicit mentions of private state can be spotted by words that mark someone's private states directly. The possible examples here are:
- “The U.S. **fears** a spill-over,” said Xirao-Nima.
- The father of two British-Israeli sisters killed in a shooting in the occupied West Bank embraced their bodies while mourners sang songs of **grief** at their funeral on Sunday.
- On Saturday defence officials in Taipei **accused** Beijing of using President Tsai's US visit as an "excuse to conduct military exercises, which has seriously undermined peace, stability and security in the region".

In these examples, words "fears", "grief" and "accused" directly point to the private states of fear, grief and accusation respectively. Therefore, they are explicit mentions of private states.

Expressive subjective elements are words that do not indicate the private state directly, but rather point to it implicitly. Here are several examples of them:
- “The report is **full of absurdities**,” Xirao-Nima said.
- The father of two British-Israeli sisters killed in a shooting in the occupied West Bank embraced their bodies while **mourners** sang songs of grief at their funeral on Sunday.
- On Saturday defence officials in Taipei accused Beijing of using President Tsai's US visit as an "**excuse to conduct** military exercises, which has **seriously undermined** peace, stability and security in the region".
- **The time has come**, **gentlemen**, for Sharon, **the assassin**, to realize that **injustice cannot last long**.

Here, the expressions "full of absurdities", "mourners", and the other bold expressions implicitly indicate the private states.

Speech events are a special case of detecting subjectivity, as the word that is the anchor to the speech may itself mark subjectivity. The anchor words such as "said" or "mentioned" in most cases are neutral and don't indicate the private state, but the words such as "blame" may indicate private states. Here are some examples:
- Sargeant O’Leary **said** the incident took place at 2:00pm.
- In January, Prime Minister Rishi Sunak **called** delayed discharge "the number one problem" facing the NHS.

Here, in the first sentence the anchor word is "said", which does not indicate any subjectivity, and the even that the incident took place at 2:00pm is presented as an objective fact. In the second sentence, the word "called" indicates Rishi Sunak's subjective opinion. Note that the words "call" and "say" may or may not indicate subjectivity in different contexts. Also, speech here refers both to speaking and writing.

The sentences that do not contain any markers mentioned above and that present facts are considered objective. Note that these facts are not necessarily correct. Some examples:
- The Earth is flat.
- Their mother, Leah, is in a critical condition following surgery.

Two other things we need to mention before describing the exact task:
- There are no fixed rules about how particular words should be annotated. The instructions describe the annotations of specific examples, but do not state that specific words should always be annotated a certain way.
- Sentences should be interpreted with respect to the contexts in which they appear. The annotators should not take sentences out of context and think
what they could mean, but rather should judge them as they are being used in that particular sentence and document.
- Obviously, there are sentences which contain both objective and subjective parts. We say a sentence is subjective if it contains any subjectivity. Thus, a sentence "The Earth is undeniably spherical" is considered subjective.

## Task
You will be consequently given sentences from a newspaper. Every sentence will be surrounded by up to 5 previous and next sentences from the newspaper to provide context. (In case sentence is closer to the beginning or to the end that 5 sentences, you will be given less sentences as a context).

Your task is to assign each sentence a subjectivity score. The score will be measured on a discrete scale from 1 to 5 (also called Likert scale). You will also be given an option to assign an "Not Applicable" label. Here are the explanations of the scale:
- Scores 1 and 5 should be assigned when you are confident that the sentence is objective or subjective.
- Scores 2 and 4 should be assigned when you are unconfident but suspect that the sentence is objective or subjective.
- Score 3 should be assigned if you the instruction covers the sentence but it is difficult to say whether the sentence is objective or subjective. It is somewhere in between.
- "Not Applicable" label is used when the instruction does not cover the sentence or it is unclear how to use it.

Note that since scraping the data contains some amount of noise, you should put the "Not Applicable" label if the sentence fully consists of noise. Other cases would be questions or very short sentences, but as usual, this is quite context-dependent. Sometimes the sentences aren't cut well (like in this sentence. Here the algorithm would cut a sentence on a dot), in this case you should also put the "Not Applicable" label.