
##
## Write
##

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

def dict_to_table(_dict):
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
    dict_table = ""
    head = _dict["head"]
    dict_table += "|"+"|".join([str(n) for n in head])+"|\n"
    dict_table += "|"+"".join(["-|" for i in range(len(head))])+"\n"
    body = _dict["body"]
    for row in body:
        dict_table += "|"+"|".join([str(n) for n in row])+"|\n"

    return dict_table

##
## Read
##

def check_table(file_content_list, idx):
    return _check_table_column_name(file_content_list, idx)

def get_table(file_content_list, idx):
    """
    Get a table. If row of idx is column names of the table, this static method gets the table and row length of the table.

    Parameters
    ----------
    file_content_list: List[str]
        content list
    idx: int
        now index
    
    Returns
    -------
    table_content: List[List[str]]
        If row of idx is column names of the table, this static method return table content.
    row_length: int
        row length of the table
    """
    table_content = {}
    row_length = 0

    table_content["head"] = get_table_content(file_content_list[idx])
    table_content["body"] = []

    # column_len = len(file_content_list[i+1].split("|"))
    # idx+2 : skip column name and hyphens
    body_idx = idx + 2
    for j in range(len(file_content_list)-(body_idx)):
        if _check_table_vertical_bar(file_content_list, body_idx+j):
            row_content = get_table_content(file_content_list[body_idx+j])
            # if len(row_content) == column_len:
            table_content["body"].append(row_content)
        else:
            break
    row_length = j + 2

    return table_content, row_length

def get_table_content(row_str):
    row_content = row_str.split("|")[1:-1]
    return row_content

def _check_table_column_len(file_content_list, idx):
    return len(file_content_list[idx]) >= 3

def _check_table_vertical_bar(file_content_list, idx):
    if _check_table_column_len(file_content_list, idx):
        return file_content_list[idx][0] == "|" and file_content_list[idx][-1] == "|"
    else:
        return False

def _check_table_hyphen(file_content_list, idx):
    if _check_table_column_len(file_content_list, idx):
        return file_content_list[idx][1] == "-" and file_content_list[idx][-2] == "-"
    else:
        return False

def _check_table_column_name(file_content_list, idx):
    vb_0 = _check_table_vertical_bar(file_content_list, idx)
    if idx+1 < len(file_content_list):
        vb_1 = _check_table_vertical_bar(file_content_list, idx+1)
        h_1 = _check_table_hyphen(file_content_list, idx+1)
    else:
        vb_1 = False
        h_1 = False
    return vb_0 and vb_1 and h_1

def read(file_path):
    with open(file_path, "r") as f:
        file_content = f.read()
        file_content_list = file_content.split("\n")
        content_list = []
        file_content_list_len = len(file_content_list)
        idx = 0
        # for i in range(file_content_list_len):
        while(idx < file_content_list_len):
            content = None
            # check table
            if check_table(file_content_list, idx):
                content, row_range = get_table(file_content_list, idx)
                idx += row_range
            else:
                content = file_content_list[idx]
            content_list.append(content)
            idx += 1

    return content_list

