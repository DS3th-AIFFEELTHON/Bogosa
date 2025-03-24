from autorag.chunker import Chunker

def main():
  chunker = Chunker.from_parquet(parsed_data_path='./parsed_parquet_250310.parquet', project_dir='./chunk_project_dir_250310')
  chunker.start_chunking('chunk_config.yaml')

if __name__ == '__main__':
  main()