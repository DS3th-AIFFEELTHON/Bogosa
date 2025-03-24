from autorag.parser import Parser
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv

load_dotenv()

def main():
  parser = Parser(data_path_glob='./text/*', project_dir='./parse_project_dir')
  parser.start_parsing('./parse_config.yaml')

if __name__ == '__main__':
  main()