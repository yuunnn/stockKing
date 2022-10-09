CREATE TABLE ma60m(
         id integer PRIMARY KEY AUTOINCREMENT,
         stock_code VARCHAR,
         open_price DOUBLE,
         high_price DOUBLE,
         low_price DOUBLE,
         close_price DOUBLE,
         period_volume BIGINT,
         datetime VARCHAR
      );