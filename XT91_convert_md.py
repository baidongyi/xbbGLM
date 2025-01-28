from spire.doc import Document, FileFormat, IParagraphStyle, Stream


def convert_file(source_file_path:str, dest_file_path:str):

    # 创建文档实例
    doc = Document()

    # 加载Markdown文件
    # 从文件加载
    doc.LoadFromFile(source_file_path, FileFormat.Markdown)
    # 从字节流加载
    # doc.LoadFromStream(Stream: stream, FileFormat.Markdown)

    # 将Markdown文件转换为Word文档并保存
    doc.SaveToFile(dest_file_path, FileFormat.Docx)

    # 转换并保存为字节流
    # stream = Stream()
    # doc.SaveToStream(stream, FileFormat.Docx)
    # wordBytes = stream.ToArray()

    # 释放资源
    doc.Dispose()


if __name__ == '__main__':
    src_file = r"./readme.md"
    des_file = r"./readme.docx"
    convert_file(src_file,des_file)