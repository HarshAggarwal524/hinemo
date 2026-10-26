We investigate how multilingual transformer encoders internally reorganize their representations
when fine-tuned on code-mixed Hinglish emotion recognition. We introduce HinEmo, a
46,174-sample Hindi-English code-mixed emotion dataset constructed from four sources and
labeled via GPT-4o-mini (Cohen's κ=0.570), alongside the Hindi Lexical Dominance Score (λ), a
continuous per-sample measure of Hindi dominance derived from a corpus-level 4,318-word
Romanized Hindi lexicon. Using layer-wise linear probing, Centered Kernel Alignment, and a
controlled layer-freezing experiment across three multilingual encoders (mBERT, MuRIL,
XLM-R), we identify a fine-tuning-induced division of labor: language identity encoding shifts to
earlier layers while emotion understanding finalizes at the final layer, simultaneously and in
opposite directions. In XLM-R, this manifests as an 11-layer inversion of the Language
Specialization Layer. The degree of reorganization scales with model architecture and
correlates with downstream task performance. Freezing the identified language specialization
layers impairs XLM-R fine-tuning relative to freezing matched middle layers, providing causal
evidence that early-layer language encoding is functionally necessary in the model with the
sharpest representational reorganization. After fine-tuning, emotion representations become
largely language-robust: Hindi-dominant and English-dominant sentences reach equivalent
emotion classification depth across all three models.
