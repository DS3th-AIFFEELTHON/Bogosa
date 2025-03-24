import pandas as pd
from llama_index.llms.openai import OpenAI

from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt,
)
from autorag.data.qa.schema import Raw, Corpus
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from dotenv import load_dotenv
import os
directory_path = './chunk_project_dir_250310'

load_dotenv()

def main():
  llm = OpenAI(model='gpt-4o-mini')
  raw_df = pd.read_parquet("./parsed_parquet_250310.parquet")
  raw_instance = Raw(raw_df)

  corpus_df = pd.read_parquet("./chunk_project_dir_250310/0.parquet")
  corpus_instance = Corpus(corpus_df, raw_instance)

  initial_qa = (
      corpus_instance.sample(random_single_hop, n=70, )
      .map(
          lambda df: df.reset_index(drop=True),
      )
      .make_retrieval_gt_contents()
      .batch_apply(
          factoid_query_gen,  # query generation
          llm=llm,
          lang='ko'
      )
      .batch_apply(
          make_basic_gen_gt,  # answer generation (basic)
          llm=llm,
          lang='ko'
      )
      .batch_apply(
          make_concise_gen_gt,  # answer generation (concise)
          llm=llm,
          lang='ko'
      )
      .filter(
          dontknow_filter_rule_based,  # filter don't know
          lang="ko",
      )
  )

  initial_qa.to_parquet('./qa_250310_1.parquet', './corpus_250310_1.parquet')

if __name__ == '__main__':
  main()