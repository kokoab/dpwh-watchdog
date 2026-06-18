import unittest

from features.chat.presenters import (
    NEXT_STEP_QUESTION,
    _build_structured_contract_detail_reply,
)


class ContractDetailPresenterTests(unittest.TestCase):
    def test_detail_reply_uses_shared_next_step_question(self):
        reply = _build_structured_contract_detail_reply(
            {
                "displayed_sources": [
                    {
                        "contractId": "23IC0038",
                        "description": "Flood control project",
                        "contractor": "JGO BUILDERS",
                        "category": "Flood Control",
                        "status": "Completed",
                        "budget": 48509603.99,
                        "progress": 100,
                    }
                ]
            }
        )

        self.assertIn("Flood control project (23IC0038)", reply)
        self.assertIn(NEXT_STEP_QUESTION.strip(), reply)


if __name__ == "__main__":
    unittest.main()
