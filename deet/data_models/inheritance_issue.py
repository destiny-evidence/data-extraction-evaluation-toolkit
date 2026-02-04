"""Demonstrates the inheritance problem we have."""

from loguru import logger
from pydantic import BaseModel


class Attribute(BaseModel):
    """A base attribute."""

    attribute_id: int


class EppiAttribute(Attribute):
    """An Eppi extension of the attribute."""

    eppi_attribute_id: int


class AttributeList(BaseModel):
    """A list of base attributes."""

    attributes: list[Attribute]

    def print_attributes(self) -> None:
        """Print the attribute id of each attribute."""
        for att in self.attributes:
            logger.info(att.attribute_id)


class EppiAttributeList(AttributeList):
    """
    A list of eppi attributes should extend the base list,
    but we cannot override attributes.
    """

    attributes: list[EppiAttribute]

    def print_eppi_attributes(self) -> None:
        """Print the eppi attribute id of each attribute."""
        for att in self.attributes:
            logger.info(att.eppi_attribute_id)


# Now if we define an add_attribute function
def add_attribute(attr_list: AttributeList, attr: Attribute) -> None:
    """Add an attribute to a list of attributes."""
    attr_list.attributes.append(attr)


# Mypy doesn't have a problem with letting us add an attribute to an
# EppiAttributeList
eppi_list = EppiAttributeList(attributes=[])
add_attribute(eppi_list, Attribute(attribute_id=123))

# And so this will now fail
eppi_list.print_eppi_attributes()
