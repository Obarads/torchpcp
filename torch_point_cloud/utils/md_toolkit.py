
def dict2mdtables(tables, file_name, open_mode="w"):
    """
    Examples
    --------
    {
        "Result 1":{
            "head":["PC","Mac","Windows","Ubuntu"],
            "body":[
                ["X1","x","o","o"],
                ["Surface","x","o","o"],
                ["MacBook Air","o","o","o"]
            ],
        },
        "Result 2":{
            ...
        }
    }
    """
    with open(file_name, mode=open_mode) as f:
        for key in tables:
            f.write("\n{}\n\n".format(key))
            create_markdown_table(f, tables[key])
        f.close()

def dict2mdtable(table, file_name, open_mode="w"):
    """
    Examples
    --------
    {
        "head":["PC","Mac","Windows","Ubuntu"],
        "body":[
            ["X1","x","o","o"],
            ["Surface","x","o","o"],
            ["MacBook Air","o","o","o"]
        ]
    }
    """
    with open(file_name, mode=open_mode) as f:
       create_markdown_table(f, table)
       f.close()

def create_markdown_table(f, table):
    """
    Examples
    --------
    {
        "head":["PC","Mac","Windows","Ubuntu"],
        "body":[
            ["X1","x","o","o"],
            ["Surface","x","o","o"],
            ["MacBook Air","o","o","o"]
        ]
    }
    """
    head = table["head"]
    f.write("|"+"|".join([str(n) for n in head])+"|\n")
    f.write("|"+"".join(["-|" for i in range(len(head))])+"\n")
    body = table["body"]
    for row in body:
        f.write("|"+"|".join([str(n) for n in row])+"|\n")


