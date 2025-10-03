from typing import Annotated, List, TypedDict, Union

class State_ax(TypedDict):
    images: Annotated[List[Union[str, bytes]], "List of image paths, bytes, or file uploads"]
    primary: Annotated[List[Union[str, bytes]], "Output of primary"]
    anomaly: Annotated[List[Union[str, bytes]], "Output of anomaly"]
    quantity: Annotated[List[str], "output quantity"]
    bn_ed : Annotated[List[str], "output batch number and expired date"]
    url: Annotated[List[str], "List of image URLs"]
    po_number: Annotated[List[str], "PO number of the invoice/faktur"]