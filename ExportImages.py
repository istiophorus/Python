import pypyodbc as po

def save_data_to_file(file_name, data):
    with open(file_name, "wb") as ff:
        ff.write(data)


targetDir = "d:/files/"
connection = po.connect(r'Driver={SQL Server};Server=PL-L-7005231\BORNHOLM;Database=Domino_2.0;Trusted_Connection=yes;')
cursor = connection.cursor()
res = cursor.execute("SELECT cast(Id as varchar(36)), Data, [Format] from [Bin].[ImageData] WHERE [Size] IS NOT NULL AND [Size] > 500000")

while 1:
    row = res.fetchone()
    if not row:
        break

    targetFilePath = targetDir + row[0] + "." + row[2]
    print("Writing file " + targetFilePath)

    save_data_to_file(targetFilePath, row[1])

connection.close()