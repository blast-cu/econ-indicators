import json
import sqliteschema

from data_utils.model_utils import dataset as d


def main():
    extractor = sqliteschema.SQLiteSchemaExtractor(d.DB_FILENAME)
    # print(
    #     "--- dump all of the table schemas into a dictionary ---\n{}\n".format(
    #         json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)
    #     )
    # )
    print(extractor.dumps(output_format="markdown", verbosity_level=1))


if __name__ == "__main__":
    main() 