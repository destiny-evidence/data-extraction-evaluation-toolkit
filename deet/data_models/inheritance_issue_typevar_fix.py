"""Demonstrates how typevar fixes the inheritance problem we have."""

from typing import Generic, TypeVar

from loguru import logger
from pydantic import BaseModel


class Attribute(BaseModel):
    """A base attribute."""

    attribute_id: int


AttributeTypeVar = TypeVar("AttributeTypeVar", bound=Attribute)


class EppiAttribute(Attribute):
    """An Eppi extension of the attribute."""

    eppi_attribute_id: int


class AttributeList(BaseModel, Generic[AttributeTypeVar]):
    """
    A list of base attributes. Attributes must be of type
    AttributeTypeVar, which defaults to base Attribute.
    """

    attributes: list[AttributeTypeVar]

    def print_attributes(self) -> None:
        """Print the attribute id of each attribute."""
        for att in self.attributes:
            logger.info(att.attribute_id)


class EppiAttributeList(AttributeList[EppiAttribute]):
    """
    Extends AttributeList but specifies that the attributes
    are of type EppiAttribute.
    """

    def print_eppi_attributes(self) -> None:
        """Print the eppi attribute id of each attribute."""
        for att in self.attributes:
            logger.info(att.eppi_attribute_id)


# We can use the typevar to add an attribute of any type to any
# type of list of attributes
def add_attribute(
    attr_list: AttributeList[AttributeTypeVar], attr: AttributeTypeVar
) -> None:
    """Add an attribute to a list of attributes."""
    attr_list.attributes.append(attr)


eppi_list = EppiAttributeList(attributes=[])

# Now, if we try to add a base attribute to a list of Eppi Attributes,
# mypy will complain
add_attribute(eppi_list, Attribute(attribute_id=123))
add_attribute(eppi_list, EppiAttribute(attribute_id=123, eppi_attribute_id=1234))


# This will now fail at runtime
eppi_list.print_eppi_attributes()
