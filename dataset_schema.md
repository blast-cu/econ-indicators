# user
|  Field   |     Type     | Nullable | Key | Default | Index | Extra |
| -------- | ------------ | -------- | --- | ------- | :---: | ----- |
| id       | INTEGER      | NO       |     |         |       |       |
| email    | VARCHAR(100) | YES      |     | NULL    |       |       |
| password | VARCHAR(100) | YES      |     | NULL    |       |       |
| admin    | BOOLEAN      | NO       |     |         |       |       |
| PRIMARY  | KEY          | YES      |     | NULL    |       |       |
| UNIQUE   | (email)      | YES      |     | NULL    |       |       |

# topic
|    Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ----------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id          | INTEGER       | NO       |     |         |       |       |
| description | VARCHAR(1000) | YES      |     | NULL    |       |       |
| relevant    | BOOLEAN       | NO       |     |         |       |       |
| name        | VARCHAR(1000) | NO       |     |         |       |       |
| PRIMARY     | KEY           | YES      |     | NULL    |       |       |
| UNIQUE      | (name)        | YES      |     | NULL    |       |       |

# cluster
|    Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ----------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id          | INTEGER       | NO       |     |         |       |       |
| name        | VARCHAR(1000) | NO       |     |         |       |       |
| explanation | VARCHAR(1000) | YES      |     | NULL    |       |       |
| PRIMARY     | KEY           | YES      |     | NULL    |       |       |
| UNIQUE      | (name)        | YES      |     | NULL    |       |       |

# clusters
|   Field    |  Type   | Nullable | Key | Default | Index | Extra |
| ---------- | ------- | -------- | --- | ------- | :---: | ----- |
| cluster_id | INTEGER | NO       |     |         |       |       |
| user_id    | INTEGER | NO       |     |         |       |       |
| PRIMARY    | KEY     | YES      |     | NULL    |       |       |
| user_id)   |         | YES      |     | NULL    |       |       |

# article
|    Field     |     Type      | Nullable | Key | Default | Index | Extra |
| ------------ | ------------- | -------- | --- | ------- | :---: | ----- |
| id           | INTEGER       | NO       |     |         |       |       |
| headline     | VARCHAR(1000) | NO       |     |         |       |       |
| source       | VARCHAR(100)  | NO       |     |         |       |       |
| keywords     | VARCHAR(100)  | NO       |     |         |       |       |
| num_keywords | INTEGER       | NO       |     |         |       |       |
| relevance    | FLOAT         | NO       |     |         |       |       |
| text         | VARCHAR(5000) | NO       |     |         |       |       |
| distance     | FLOAT         | YES      |     | NULL    |       |       |
| date         | DATETIME      | NO       |     |         |       |       |
| url          | VARCHAR(100)  | NO       |     |         |       |       |
| cluster_id   | INTEGER       | YES      |     | NULL    |       |       |
| PRIMARY      | KEY           | YES      |     | NULL    |       |       |

# articleann
|    Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ----------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id          | INTEGER       | NO       |     |         |       |       |
| user_id     | INTEGER       | NO       |     |         |       |       |
| article_id  | INTEGER       | NO       |     |         |       |       |
| frame       | VARCHAR       | YES      |     | NULL    |       |       |
| econ_rate   | VARCHAR       | YES      |     | NULL    |       |       |
| econ_change | VARCHAR       | YES      |     | NULL    |       |       |
| comments    | VARCHAR(1000) | YES      |     | NULL    |       |       |
| text        | VARCHAR(5000) | YES      |     | NULL    |       |       |
| PRIMARY     | KEY           | YES      |     | NULL    |       |       |

# quantity
|   Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ---------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id         | VARCHAR(1000) | NO       |     |         |       |       |
| local_id   | INTEGER       | NO       |     |         |       |       |
| article_id | INTEGER       | NO       |     |         |       |       |
| PRIMARY    | KEY           | YES      |     | NULL    |       |       |

# topics
|     Field      |  Type   | Nullable | Key | Default | Index | Extra |
| -------------- | ------- | -------- | --- | ------- | :---: | ----- |
| topic_id       | INTEGER | NO       |     |         |       |       |
| articleann_id  | INTEGER | NO       |     |         |       |       |
| PRIMARY        | KEY     | YES      |     | NULL    |       |       |
| articleann_id) |         | YES      |     | NULL    |       |       |

# quantityann
|      Field       |     Type      | Nullable | Key | Default | Index | Extra |
| ---------------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id               | INTEGER       | NO       |     |         |       |       |
| user_id          | INTEGER       | NO       |     |         |       |       |
| quantity_id      | INTEGER       | NO       |     |         |       |       |
| type             | VARCHAR(1000) | YES      |     | NULL    |       |       |
| macro_type       | VARCHAR(1000) | YES      |     | NULL    |       |       |
| industry_type    | VARCHAR(1000) | YES      |     | NULL    |       |       |
| gov_level        | VARCHAR(1000) | YES      |     | NULL    |       |       |
| gov_type         | VARCHAR(1000) | YES      |     | NULL    |       |       |
| expenditure_type | VARCHAR(1000) | YES      |     | NULL    |       |       |
| revenue_type     | VARCHAR(1000) | YES      |     | NULL    |       |       |
| comments         | VARCHAR(1000) | YES      |     | NULL    |       |       |
| spin             | VARCHAR(1000) | YES      |     | NULL    |       |       |
| PRIMARY          | KEY           | YES      |     | NULL    |       |       |

# articles
|   Field    |  Type   | Nullable | Key | Default | Index | Extra |
| ---------- | ------- | -------- | --- | ------- | :---: | ----- |
| article_id | INTEGER | NO       |     |         |       |       |
| user_id    | INTEGER | NO       |     |         |       |       |
| PRIMARY    | KEY     | YES      |     | NULL    |       |       |
| user_id)   |         | YES      |     | NULL    |       |       |

