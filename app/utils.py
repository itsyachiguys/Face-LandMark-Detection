import xml.etree.ElementTree as ET


def load_xml(xml_file):
    """
    Load an XML annotation file.

    Returns:
        ElementTree root
    """
    tree = ET.parse(xml_file)
    return tree.getroot()