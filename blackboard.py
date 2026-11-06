import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class Blackboard:
    user_request: str #request from the user that models will solve


    #holds the actual output of each part of the job and default_factory creates a new dictionary per Blackboard (a plain default would be shared by all) 
    sections: Dict[str, Any] = field(default_factory=lambda: {
        #Each section is an empty string at first --> model fills them with outputs
        "code": "", "tests": "", "docs": "", "security": "",
    } )


    #now to track the progress of each section for other models, create a way for them to know being pending -> in progress -> completed
    status: dict[str,str] = field(default_factory=lambda: {
        #start everything as pending first
        "code":"pending","tests":"pending","docs":"pending","security":"pending"
    })


    #create a history of events for debugging! (and perhaps a ui later)
    log: List[str] = field(default_factory=list)


    #coordination for the different models w/ a lock + wait and notify system. #init=False means does not get passed in, repr=False means hide it when printing, default_factory = asyncio just creates a new fresh condition for each board.
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition, init=False, repr=False)


    #async because to grab the actual "lock" u may need to wait for another worker to release it. returns true if the caller "won" but false is someone already took it
    async def claim(self, section:str) -> bool:
        """Take a section and only return true only for the first caller."""
        async with self._cond:
            if self.status[section] != "pending":
                return False
            self.status[section] = "in_progress"
            self.log.append(f"{section}: in_progress")
            self._cond.notify_all() #notify all waiting models that a section has been claimed
            return True


    #if a model has finished and has something to put onto the board, worker calls this
    async def post(self, section:str, content:str) -> None:
        """Write the finished work onto the board and wake everyone (the other ai models) up"""
        async with self._cond:
            self.sections[section] = content
            self.status[section] = "completed"
            self.log.append(f"{section}: completed")
            self._cond.notify_all() #notify all waiting models that a section has been completed


    #lets a worker sleep until some condition about the board becomes true
    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        """Sleep until the predicate is True and wake when the board changes"""
        async with self._cond:
            #checks predicate and if false, releases the lock and sleep until notify_all trigger and then re-checks. Returns once predicate is actually True, 
            await self._cond.wait_for(predicate)


    def is_done(self) -> bool:
        """Check if all sections are completed"""
        return all(status == "completed" for status in self.status.values())#only true if every status is completed



#just a quick setup with fake models ill link models later

async def fake_model(board:Blackboard, name:str, needs:list[str],delay:float):
    await board.wait_until(lambda: all(board.status[section] == "completed" for section in needs)) #wait until all needed sections are completed

    #try to claim section, if not, just return and let another model do it
    if await board.claim(name):
        await asyncio.sleep(delay) #simulate work being done
        await board.post(name, f"{name} output") #post the output to the board



#entry point for testing
async def main():
    board = Blackboard(user_request="Please generate code, tests, docs, and security checks.")

    #create a list of fake models with their dependencies and delays
    models = [
        fake_model(board, "code", [], 1), #code has no dependencies
        fake_model(board, "tests", ["code"], 1), #tests depend on code
        fake_model(board, "docs", ["code"], 1), #docs depend on code
        fake_model(board, "security", ["code"], 1), #security depends on code
    ]

    await asyncio.gather(*models) #run all models concurrently

    print("Final Blackboard State:")
    print(board.sections)
    print("Log of events:")
    print(board.log)


if __name__ == "__main__":
    asyncio.run(main())