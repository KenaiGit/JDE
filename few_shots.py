few_shots = [
    {
        "Question": "What is the price of the 'Laptop 14\" i5 Gen9'?",
        "SQLQuery": "SELECT Product_Name, Unit_Price FROM Products WHERE Product_Name = 'Laptop 14\" i5 Gen9';",
        "SQLResult": "Product_Name: 'Laptop 14\" i5 Gen9', Unit_Price: '$296.36'",
        "Answer": "The price of the 'Laptop 14\" i5 Gen9' is $296.36."
    },
    {
        "Question": "What products are available in the WH-Central warehouse?",
        "SQLQuery": "SELECT Product_Name, Stock_Qty FROM Products WHERE Warehouse = 'WH-Central';",
        "SQLResult": "Product_Name: 'Laptop 14\" i5 Gen9', Stock_Qty: 10\nProduct_Name: 'External Hard Drive 1TB', Stock_Qty: 126",
        "Answer": "The following products are available in the WH-Central warehouse: Laptop 14\" i5 Gen9 (10 units), External Hard Drive 1TB (126 units)."
    },
    {
        "Question": "How many units of 'Smartphone 5G Pro' are available in stock?",
        "SQLQuery": "SELECT Stock_Qty FROM Products WHERE Product_Name = 'Smartphone 5G Pro';",
        "SQLResult": "Stock_Qty: 52",
        "Answer": "There are 52 units of 'Smartphone 5G Pro' available in stock."
    },
    {
        "Question": "What is the price of 'Wireless Mouse' in the Inventory?",
        "SQLQuery": "SELECT Product_Name, Unit_Price FROM Products WHERE Product_Name = 'Wireless Mouse';",
        "SQLResult": "Product_Name: 'Wireless Mouse', Unit_Price: '$15.99'",
        "Answer": "The price of the 'Wireless Mouse' is $15.99."
    },
    {
        "Question": "How many units of 'Portable Bluetooth Speaker' are available in the WH-East warehouse?",
        "SQLQuery": "SELECT Stock_Qty FROM Products WHERE Warehouse = 'WH-East' AND Product_Name = 'Portable Bluetooth Speaker';",
        "SQLResult": "Stock_Qty: 60",
        "Answer": "There are 60 units of 'Portable Bluetooth Speaker' available in the WH-East warehouse."
    },
    {
        "Question": "What is the price of '4K TV 55-inch'?",
        "SQLQuery": "SELECT Product_Name, Unit_Price FROM Products WHERE Product_Name = '4K TV 55-inch';",
        "SQLResult": "Product_Name: '4K TV 55-inch', Unit_Price: '$850.99'",
        "Answer": "The price of the '4K TV 55-inch' is $850.99."
    },
    {
        "Question": "What products are in the 'WH-West' warehouse?",
        "SQLQuery": "SELECT Product_Name, Stock_Qty FROM Products WHERE Warehouse = 'WH-West';",
        "SQLResult": "Product_Name: 'Laptop 14\" i5 Gen9', Stock_Qty: 15\nProduct_Name: 'External Hard Drive 1TB', Stock_Qty: 89",
        "Answer": "The following products are available in the WH-West warehouse: Laptop 14\" i5 Gen9 (15 units), External Hard Drive 1TB (89 units)."
    },
    {
        "Question": "How many 'USB-C Cable' units are available in the 'WH-South' warehouse?",
        "SQLQuery": "SELECT Stock_Qty FROM Products WHERE Warehouse = 'WH-South' AND Product_Name = 'USB-C Cable';",
        "SQLResult": "Stock_Qty: 230",
        "Answer": "There are 230 units of 'USB-C Cable' available in the WH-South warehouse."
    },
    {
        "Question": "What is the price of 'Wireless Keyboard'?",
        "SQLQuery": "SELECT Product_Name, Unit_Price FROM Products WHERE Product_Name = 'Wireless Keyboard';",
        "SQLResult": "Product_Name: 'Wireless Keyboard', Unit_Price: '$29.99'",
        "Answer": "The price of the 'Wireless Keyboard' is $29.99."
    },
    {
        "Question": "How many units of 'Laptop 14\" i7 Gen10' are available in stock?",
        "SQLQuery": "SELECT Stock_Qty FROM Products WHERE Product_Name = 'Laptop 14\" i7 Gen10';",
        "SQLResult": "Stock_Qty: 18",
        "Answer": "There are 18 units of 'Laptop 14\" i7 Gen10' available in stock."
    }
]
