from product_search_ranker.text import query_slices, strip_html, tokenize


def test_strip_html_collapses_whitespace():
    assert strip_html("<p>Hello&nbsp;<b>world</b></p>\n\n  x") == "Hello world x"
    assert strip_html(None) == ""


def test_tokenize_lowercases_strips_accents_and_keeps_numbers():
    assert tokenize("Funda iPhone 13 sin cordón", "es") == ["funda", "iphone", "13", "cordon"]
    assert tokenize("The 12 oz Cup", "us") == ["12", "oz", "cup"]


def test_length_slices():
    assert "len_1_2" in query_slices("usb cable", "us")
    assert "len_3_5" in query_slices("usb c cable fast", "us")
    assert "len_6_plus" in query_slices("running shoes men without laces size ten", "us")


def test_negation_is_per_language():
    assert "negation" in query_slices("shoes without laces", "us")
    assert "negation" not in query_slices("shoes without laces", "es")
    assert "negation" in query_slices("funda iphone sin cordón", "es")


def test_number_or_unit():
    assert "number_or_unit" in query_slices("iphone 13 case", "us")
    assert "number_or_unit" in query_slices("water 16 oz bottle", "us")
    assert "number_or_unit" not in query_slices("red running shoes", "us")


def test_brand_in_query_needs_whole_tokens():
    assert "brand_in_query" in query_slices("nike running shoes", "us", ["Nike", "Adidas"])
    assert "brand_in_query" not in query_slices("sunike shoes", "us", ["Nike"])
    assert "brand_in_query" not in query_slices("nike shoes", "us", [None, ""])
