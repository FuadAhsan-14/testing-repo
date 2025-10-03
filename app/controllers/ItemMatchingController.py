from app.generative import manager

from app.services.MatchingAgentService import MatchingAgentService
from app.schemas.AgentMatchingOutputSchema import MatchingSchema

from app.utils.Http import response_success, response_error

class MatchingController:
    def __init__(self):
        llm = manager._get_llm(name="gemini_model_flash_lite25", temperature=0)
        self.agent = MatchingAgentService(llm=llm)

    async def match_lists(self, req: MatchingSchema):
        """
        Logika utama untuk memproses permintaan pencocokan list.
        Berfungsi sebagai lapisan API yang memanggil service.
        """
        try:
            # Ekstrak list dari request body
            po_items = req.po_item_list
            faktur_items = req.faktur_item_list

            # Panggil metode execute dari agent service yang sudah diinisialisasi
            result = await self.agent.execute(
                po_items=po_items,
                faktur_items=faktur_items
            )

            # Kembalikan hasil dalam format respons sukses
            return response_success(result)

        except Exception as e:
            # Tangani error dan kembalikan format respons error
            print(f"An error occurred in MatchingController: {e}")
            raise response_error(str(e))

# Buat satu instance dari controller untuk digunakan di seluruh aplikasi
matching_controller = MatchingController()