import os
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest
from factcheck.explain import get_shap_explainer, explain_claim

def test_explain_claim_structure():
    explainer, model, tokenizer = get_shap_explainer("distilbert-base-uncased", device=-1)
    
    test_claim = "Vaccines cause autism."
    result = explain_claim(test_claim, explainer, target_class_idx=0)
    
    assert "text" in result
    assert "base_value" in result
    assert "token_attributions" in result
    assert isinstance(result["token_attributions"], list)
    assert len(result["token_attributions"]) > 0
    
    first_token = result["token_attributions"][0]
    assert "token" in first_token
    assert "score" in first_token
    assert isinstance(first_token["score"], float)