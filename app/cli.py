import asyncio
from app.models import Status
from app.orchestrator import approve, run_research


async def main() -> None:
    print("EvidenceOps Agent - type 'exit' to stop")
    while True:
        question = input("\nResearch question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        request = await run_research(question)
        if request.status == Status.FAILED:
            print(f"\nRequest failed: {request.failure_reason}")
            continue

        print("\n--- Draft ---\n")
        print(request.report)
        approval = input("\nSave an approved final report? [y/N]: ").strip().lower()
        if approval == "y":
            print(approve(request))


if __name__ == "__main__":
    asyncio.run(main())
