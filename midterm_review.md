<h2 id="generative-probabilistic-models">Generative probabilistic models</h2>
<p>We have discussed the following generative probabilistic models:</p>
<ul>
<li>Naïve Bayes classifier</li>
<li>N-gram Language Model</li>
<li>Hidden Markov Model</li>
</ul>
<p>For each of these, you should be able to</p>
<ul>
<li>describe the imagined process by which data is generated, and say what independence assumptions are made (includes familiarity with terms like &quot;Markov assumption” and “trigram model&quot;).</li>
<li>write down the associated formula for the joint probability of hidden and observed variables (or just the observed variables if there are no hidden variables).</li>
<li>compute the probability of (say) a tag-word sequence, document, or whatever the model describes (assuming you know the model parameters).</li>
<li>for the naïve Bayes model, compute the most probable class for a particular input, hand-simulating any algorithms that might be needed (again assuming you know the model parameters).</li>
<li>for the HMM, convert between a state transition diagram and transition table, and explain why a dynamic programming algorithm is needed for certain operations (decoding—Viterbi algorithm; marginalizing over tags—forward algorithm). You do not need to understand (for the midterm) how these algorithms work.</li>
<li>explain how the model is trained.</li>
<li>give examples of tasks the model could be applied to, and how it would be applied.</li>
<li>say what the model can and cannot capture about natural language, ideally giving examples of its failure modes.</li>
</ul>
<h2 id="discriminative-classification-models">Discriminative classification models</h2>
<p>We have covered:</p>
<ul>
<li>binary &amp; multiclass Perceptron</li>
<li>SVM</li>
<li>multiclass Logistic regression/MaxEnt</li>
</ul>
<p>For this model, you should be able to</p>
<ul>
<li>give examples of tasks the model could be applied to, and how it would apply (e.g., what features might be useful).</li>
<li>explain at a high level what training the model aims to achieve, and how it differs from training a generative model.</li>
<li>identify which models can be trained with early stopping, averaging, or regularization, and why these techniques are used.</li>
<li>discuss the pros and cons of discriminative classifiers vs. Naïve Bayes.</li>
<li>explain why optimization is required for learning, unlike with generative models.</li>
<li>identify MaxEnt as probabilistic and the others as non-probabilistic. You should understand the formula for computing the conditional probability of the hidden class given the observations/features, and be able to apply that formula if you are given an example problem with features and weights. You do not need to memorize the formula.</li>
<li>explain how naïve Bayes can be expressed as a linear classifier.</li>
<li>write and explain the decoding (classification) rule for any linear classifier given a vector of weights and a feature function.</li>
<li>walk through the Perceptron learning algorithm for an example.</li>
<li>explain how, for the Perceptron, decoding is embedded as a step within learning.</li>
</ul>
<h2 id="other-formulas">Other formulas</h2>
<p>In addition to the equations for the generative and discriminative models listed above, you should know the formulas for the following concepts, what they may be used for, and be able to apply them appropriately. Where relevant you should be able to discuss strengths and weaknesses of the associated method, and alternatives.</p>
<ul>
<li>Bayes&#39; Rule (also: definition of Condition Probability, law of Total Probability aka Sum Rule, Chain Rule, marginalization, and all other relevant formulas in the Basic Probability Theory reading)</li>
<li>Noisy channel model</li>
<li>Add-One / Add-Alpha Smoothing</li>
<li>Interpolation (for language model smoothing)</li>
<li>Dot product</li>
<li>Precision, recall, and F1-score</li>
</ul>
<h2 id="additional-mathematical-and-computational-concepts">Additional Mathematical and Computational Concepts</h2>
<p> Overarching concepts:</p>
<ul>
<li><p>Zipf&#39;s Law and sparse data: What is Zipf&#39;s law and what are its implications? What does &quot;sparse data&quot; refer to? Be able to discuss these with respect to specific tasks.</p>
</li>
<li><p>Probability estimation and smoothing: What are different methods for estimating probabilities from corpus data, and what are the pros and cons of each, and the characteristic errors? Under what circumstances might you find simpler methods acceptable, or unacceptable? You should be familiar at a high level at least with:</p>
<ul>
<li>Maximum Likelihood Estimation</li>
<li>Add-One / Add-Alpha Smoothing</li>
<li>Lower-order smoothing</li>
<li>Interpolation</li>
<li>Backoff</li>
<li>Good-Turing Smoothing</li>
<li>Kneser-Ney Smoothing</li>
<li>Entropy and Cross-Entropy/Perplexity</li>
</ul>
<p>Except as noted under &quot;Formulas&quot; above, you do not need to memorize the formulas, but should understand the conceptual differences and motivation behind each method, and should be able to <em>use</em> the formulas if they are given to you.</p>
</li>
<li><p>Training, development, and test sets: How are these used and for what reason? Be able to explain their application to particular problems.</p>
</li>
<li><p>Cross-validation</p>
</li>
<li><p>The distinction between parameters and hyperparameters</p>
</li>
<li><p>The distinction between models and algorithms</p>
</li>
<li><p>Objective function: Classification objective, Learning objective/Loss function</p>
</li>
</ul>
<h2 id="linguistic-and-representational-concepts">Linguistic and Representational Concepts</h2>
<p>You should be able to explain each of these concepts, give one or two examples where appropriate, and be able to identify examples if given to you. You should be able to say what NLP tasks these are relevant to and why.</p>
<ul>
<li>Ambiguity (of many varieties, w.r.t. all tasks we&#39;ve discussed)</li>
<li>Part-of-Speech<ul>
<li>especially, the terms: common noun, proper noun, pronoun, adjective, adverb, auxiliary, main verb, preposition, conjunction, punctuation</li>
<li>open-class vs. closed-class</li>
</ul>
</li>
<li>Word Senses and relations between them (synonym, antonym, hypernym, hyponym, similarity; homonymy vs. polysemy)</li>
<li>Word order typology: SVO, VSO, etc.</li>
<li>Dialect vs. language</li>
<li>Phonetics, phonology, lexicon, morphology, syntax, semantics, pragmatics, orthography</li>
<li>Synthetic vs. analytic language</li>
<li>Inflectional vs. derivational morphology</li>
<li>consonant, vowel, tone, syllable, prosody, stress</li>
<li>morpheme, affix, prefix, suffix, compound</li>
<li>International Phonetic Alphabet (what it is—not how to use it), Unicode</li>
<li>Language families, e.g. Indo-European, Romance, Germanic, Slavic, Sino-Tibetan, Semitic</li>
<li>Salient aspects of languages presented thus far (questions will be about groups of languages, so if you missed one or two presentations that shouldn’t put you at a disadvantage)</li>
</ul>
<p>Also, you should be able to give an analysis of a phrase or sentence using the following formalisms. Assume that either the example will be very simple and/or some set of labels is provided for you to use. (i.e. you should know some standard categories for English but you don&#39;t need to memorize details of specific tagsets etc.)</p>
<ul>
<li>label parts of speech</li>
<li>label word senses given dictionary definitions</li>
<li>label named entities given a list of classes</li>
<li>label the sentiment of a document as well as some positive and negative cue words</li>
</ul>
<h2 id="tasks">Tasks</h2>
<p>You should be able to explain each of these tasks, give one or two examples where appropriate, and discuss cases of ambiguity or what makes the task difficult. In most cases you should be able to say what algorithm(s) or general method(s) can be used to solve the task, and what evaluation method(s) are typically used.</p>
<ul>
<li>Tokenization</li>
<li>Spelling correction</li>
<li>Language modeling</li>
<li>PoS-tagging</li>
<li>Text categorization</li>
<li>Word sense disambiguation</li>
<li>Named entity recognition</li>
<li>Sentiment analysis</li>
</ul>
<h2 id="corpora-resources-and-evaluation">Corpora, Resources, and Evaluation</h2>
<p>You should be able to describe what linguistic information is captured in each of the following resources, and how it might be used in an NLP system.</p>
<ul>
<li>Penn Treebank</li>
<li>WordNet</li>
</ul>
<p>For each of the following evaluation measures, you should be able to explain what it measures, what tasks it would be appropriate for, and why.</p>
<ul>
<li>Cross-entropy/Perplexity</li>
<li>Accuracy</li>
<li>Precision, recall, and F1-score</li>
</ul>
<p>In addition:</p>
<ul>
<li>Intrinsic vs. extrinsic evaluation: be able to explain the difference and give examples of each for particular tasks.</li>
<li>Gold standard: what is it and what is it used for?</li>
<li>Confusion matrix: what is it and what is it used for?</li>
</ul>
<h2 id="text-processing">Text Processing</h2>
<p>You should be able to write and interpret Python-style regular expressions with the following components:</p>
<ul>
<li>string delimiters: <code>^</code> <code>$</code></li>
<li>optionality/repetition operators: <code>|</code> <code>?</code> <code>*</code> <code>+</code> <code>{3,5}</code></li>
<li>the <code>.</code> operator (any character)</li>
<li>character classes: e.g. <code>[xyz]</code>, <code>[^a-z0-9]</code> and the abbreviations <code>\w</code> <code>\W</code> <code>\s</code> <code>\S</code> <code>\d</code> <code>\D</code></li>
<li>groups: e.g. <code>([a-z][0-9])+</code></li>
<li>backslash-escaping for metacharacters</li>
</ul>
<p>You should be familiar with the Unix text commands covered in class, including the concept of piping commands together and writing to stdout or redirecting output to files.</p>
<p>You should be familiar with basic Python functionality, esp. involving strings and data structures of the types: list, tuple, dict, Counter.</p>
<p>You will not be asked to write Python code or Unix commands from scratch, but you may be asked to choose which of several commands performs the desired function, for example.</p>
<p>You should be familiar with the file formats: TSV, JSON</p>
<p>You should be familiar with the concept of version control and its benefits. We will not test you on specific version control systems or commands.</p></body>
