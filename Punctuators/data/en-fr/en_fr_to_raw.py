import os
import codecs
import xml.etree.ElementTree as etree
import re


source_dir = os.path.join('.', 'en-fr')
target_dir = os.path.join('.', 'raw')


def read_and_write_train_data(filename, output_path):
    train_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('<'):
                continue
            train_data.append(line.strip())
    with codecs.open(output_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(train_data))


def read_and_write_xml_data(filenames, output_path):
    data = []
    for filename in filenames:
        parser = etree.XMLParser(encoding='utf-8')
        root = etree.parse(filename, parser=parser).getroot().find('srcset')
        for doc in root:
            segs = doc.findall('seg')
            for seg in segs:
                data.append(seg.text.strip())
    with codecs.open(output_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(data))


def read_and_write_dev_test_data(filenames, output_path):
    cleaner = re.compile('<.*?>')
    data = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('<seg id='):
                    line = re.sub(cleaner, '', line).strip()
                    data.append(line)
    with codecs.open(output_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(data))


def main():
    # train_file = os.path.join(source_dir, 'train.tags.en-fr.en')
    # train_out = os.path.join(target_dir, 'raw.train.txt')

    ref = os.path.join('./en-fr', 'IWSLT12.TALK.dev2010.en-fr.en.xml')  # Ref
    dev_1 = os.path.join('./en-fr', 'IWSLT12.TALK.tst2010.en-fr.en.xml')
    test_1 = os.path.join('./en-fr', 'IWSLT12.TED.MT.tst2011.en-fr.en.xml')
    test_2 = os.path.join('./en-fr', 'IWSLT12.TED.MT.tst2012.en-fr.en.xml')

    # dev_files = [os.path.join(source_dir, 'IWSLT12.TALK.dev2010.en-fr.en.xml'),
    #             os.path.join(source_dir, 'IWSLT12.TALK.tst2010.en-fr.en.xml')]
    # dev_out = os.path.join(target_dir, 'raw.dev.txt')

    # test_files = [os.path.join(source_dir, 'IWSLT12.TED.MT.tst2011.en-fr.en.xml'),
    #              os.path.join(source_dir, 'IWSLT12.TED.MT.tst2012.en-fr.en.xml')]
    # test_out = os.path.join(target_dir, 'raw.test.txt')

    # read_and_write_train_data(train_file, train_out)
    read_and_write_dev_test_data([ref], os.path.join('./raw_test', 'raw.ref.txt'))
    read_and_write_dev_test_data([dev_1], os.path.join('./raw_test', 'raw.test.1.txt'))
    read_and_write_dev_test_data([test_1], os.path.join('./raw_test', 'raw.test.2.txt'))
    read_and_write_dev_test_data([test_2], os.path.join('./raw_test', 'raw.test.3.txt'))


if __name__ == '__main__':
    main()

