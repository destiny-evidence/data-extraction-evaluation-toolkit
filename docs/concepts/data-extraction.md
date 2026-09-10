# Data Extraction

The following section sets out the core components of how data extraction
is conceptualised in `deet`

## Documents

Documents are the individual units of text from which we extract data.
`deet` is designed to extract data from scientific papers,
but in principle any text can be represented as a document
(Using the CLI, this is currently limited to any pdf, or any scientific paper with an abstract).

In our data model [deet.data_models.documents.Document](../reference/api.md#deet.data_models.documents.Document),
documents must have a numeric ID, and a name.
Since we are often dealing with messy pdfs that have to be parsed,
the content field `parsed_document` is used to store the results of document parsing in text form.

Thus

```python
from deet.data_models.documents import Document, ParsedOutput

documents = [
    Document(
        id=12345678,
        name="document_1",
        parsed_document = ParsedOutput(
            text="Document 1 content",
            parser="unknown"
        )
    ),
    Document(
        id=87654321,
        name="document_2",
        parsed_document = ParsedOutput(
            text="Document 2 content",
            parser="unknown"
        )
    )
]
```

represent a minimal set of documents which could be passed to the data extractor.

## Attributes

Attributes represent the concepts we want to extract from documents, such as effect sizes, study characteristics, or any other labels applied to the documents on the basis of their content.
Each attribute must have an `attribute_label` and an `attribute_id` to identify it.
To be used for data extraction, each attribute should have a `prompt` defined.

Finally, attributes must have one of the following [`deet.data_models.base.AttributeType`](../reference/api.md#deet.data_models.base.AttributeType)s

### bool

Boolean attributes are used to represent any attribute that can be either True or False.
This could be, for example, whether a study is eligible for a systematic review (e.g. screening),
or whether a study has a particular characteristic, or can be categorised with a particular element of a taxonomy.
The following represent valid boolean attributes:

```python
from deet.data_models.base import Attribute, AttributeType

attributes = [
    Attribute(
        attribute_id=123,
        attribute_label="relevant",
        prompt="Does the article discuss the effects of climate change on human health?",
        output_data_type=AttributeType.BOOL
    ),
    Attribute(
        attribute_id=321,
        attribute_label="RCT",
        prompt="Is the article a randomised control trial?",
        output_data_type=AttributeType.BOOL
    )
]
```

### string

String attributes describe data extraction elements that can be represented as texts. For example, a string attribute could be used to extract the location of a study.

Default evaluation uses exact-match `accuracy` on string `output_data`, plus
`edit_distance_match_rate` for near matches; see
[Evaluation](evaluation.md#metrics-by-attribute-type).

### float

Float attributes describe any type of numeric data extraction elements, such as the average age of study participants, or the effect size or standard error.

Default evaluation uses exact-match `accuracy`, plus magnitude-of-error metrics
(MAE / MAPE); see [Evaluation](evaluation.md#metrics-by-attribute-type).

### integer

Integer attributes describe the subset of numeric data extraction elements that can be represented by whole numbers, and whole numbers only, for example, the number of participants in a trial, or the year in which a trial was carried out.

As with floats, default evaluation uses exact-match `accuracy`, plus MAE / MAPE;
see [Evaluation](evaluation.md#metrics-by-attribute-type).

### list

Lists describe data extraction elements that have 0, 1, or more elements. For example, the prompt "extract all of the health outcomes described in the study" could be represented as a list

!!! Warning "Not fully supported"
    List attributes are not covered by currently implemented standard evaluation metrics

### dict

dict (dictionary) attributes describe anything that can be represented by key value pairs. This could be used, for example to extract and parse tables from a study:

```python
table_dictionary = {
    "effect_size": 0.1,
    "standard error": 0.02,
    "intervention name": "Paracetemol"
}
```

!!! Warning "Not fully supported"
    dict attributes are not reliably parsed from EppiJson, and are not covered by currently implemented standard evaluation metrics.

## Annotations

Annotations describe the value (either according to the gold standard, or as predicted by an LLM) of an attribute for a document. Thus

```python
from deet.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
)

attribute_relevance = Attribute(
    attribute_id=123,
    attribute_label="relevant",
    prompt="Does the article discuss the effects of climate change on human health?",
    output_data_type=AttributeType.BOOL,
)

annotation = GoldStandardAnnotation(
    attribute=attribute_relevance,
    raw_data=True,
    annotation_type=AnnotationType.HUMAN,
)
```

represents a decision by a human that a document was relevant, according to the attribute with id=123.

## Annotated documents

Human or LLM annotations are attached to a document through a
[´deet.data_models.documents.GoldStandardAnnotatedDocument`](../reference/api.md#deet.data_models.documents.GoldStandardAnnotatedDocument). Thus

```python
from deet.data_models.documents import GoldStandardAnnotatedDocument

document = Document(
    id=12345678,
    name="document 1",
    ...
)

gold_standard_document = GoldStandardAnnotatedDocument(
    document = document,
    annotations = [annotation]
)
```

represents the fact that the document named "document 1" had been labelled as relevant by a human.

## Extraction methods

Once we have documents and attributes, `deet` needs to work out the value of
each attribute for each document. How it does this is controlled by the `method`
field on [`deet.extractors.base_extractor.DataExtractionConfig`](../reference/api.md#deet.extractors.base_extractor.DataExtractionConfig),
which takes one of the [`deet.extractors.base_extractor.ExtractionMethod`](../reference/api.md#deet.extractors.base_extractor.ExtractionMethod)s.
The method we choose also changes how an attribute's `prompt` is interpreted.

### LLM (default)

The default method sends the document context and the list of attributes to an
LLM, and asks it to return a structured answer for every attribute. Here the
`prompt` is a question or instruction written in natural language, for example
"Does the article discuss the effects of climate change on human health?". This
is the only method that produces values for every attribute type, and the LLM is
chosen with the `provider` and `model` fields.

### Keyword

The keyword method does no LLM call at all: it simply checks whether the words we
are looking for appear in the document. Here the `prompt` is not a question but a
list of keyphrases, separated by semicolons (`;`). An attribute is marked `True`
if any of its phrases appears in the document, and matching is case-insensitive.
Thus

```python
Attribute(
    attribute_id=123,
    attribute_label="relevant",
    prompt="climate change; global warming; greenhouse gas",
    output_data_type=AttributeType.BOOL,
)
```

marks a document as relevant if it mentions any of those three phrases. The keyword method only supports boolean attributes such as screening or categorisation.

### Semantic

The semantic method works like the keyword method, but matches on meaning rather
than exact strings. We split the document into sentences and use a
sentence-transformer model to compare each of the attribute's phrases against
each sentence. An attribute is marked True if the highest similarity between
any phrase and any sentence is at least semantic_similarity_threshold (0.5 by
default).

As with the keyword method, the prompt is a semicolon-separated list of phrases,
and only True matches are produced. For this method the model field must name
a [sentence-transformer model](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) rather than an LLM:

```yaml
method: semantic
model: sentence-transformers/all-MiniLM-L6-v2
semantic_similarity_threshold: 0.5
```

!!! Note "Tuning the threshold"
    Short phrases compared against whole sentences rarely score very highly, so
    the right threshold depends on the model and documents. If the semantic
    method returns False for everything, try lowering
    semantic_similarity_threshold.
