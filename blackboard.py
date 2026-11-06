import asyncio
import importlib.util
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama") #my local Ollama server so no api key needed

limiter = asyncio.Semaphore(1)

@dataclass
class Blackboard:
    user_request: str #request from the user that models will solve


    #holds the actual output of each part of the job and default_factory creates a new dictionary per Blackboard (a plain default would be shared by all) 
    sections: Dict[str, Any] = field(default_factory=lambda: {
        #Each section is an empty string at first --> model fills them with outputs
        "code": "", "tests": "", "docs": "", "security": "", "verity": "",
    } )


    #now to track the progress of each section for other models, create a way for them to know being pending -> in progress -> completed
    status: dict[str,str] = field(default_factory=lambda: {
        #start everything as pending first
        "code":"pending","tests":"pending","docs":"pending","security":"pending","verity":"pending"
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


#configuring models

ROLES = {
    "code":{
        "model": "llama3.2:3b",
        "needs": [],
        "prompt": "Write a complete solution in Python 3 for the following task. Put the final code in a single ```python fenced code block. Do not include tests or example usage. Keep it short. \n\nTask: {request}"
    },
    "tests":{
        "model": "llama3.2:3b",
        "needs": ["code"],
        "prompt": "Write a few short unit tests for this Python code using the built-in unittest module. Assume the code is already defined above your tests, so do NOT import it from anywhere and do NOT copy it. Put the tests in a single ```python fenced code block:\n\n{code}"
    },
    "docs":{
        "model": "llama3.2:3b",
        "needs": ["code"],
        "prompt": "Write brief documentation for this Python code using Google-style docstrings:\n\n{code}"
    },
    "security":{
        "model": "llama3.2:3b",
        "needs": ["code"],
        "prompt": "You are a code reviewer. List any bugs, bad input handling, or edge cases in this Python code, in a few short bullet points:\n\n{code}"
    },
}


async def model_worker(board: Blackboard, name: str):
    cfg = ROLES[name]#workers settings

    #sleep until every section in "needs" is completed
    await board.wait_until(lambda: all(board.status[need] == "completed" for need in cfg["needs"]))

    #try to claim the section, if someone else already did, just return
    if await board.claim(name):
        #build the prompt for the model
        prompt = cfg["prompt"].format(request=board.user_request, code=board.sections.get("code", ""))

        try:
            async with limiter:
                print(f"[{name}] working...") 
                #call the model and get the response
                response = await client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
            #post the response to the board
            await board.post(name, response.choices[0].message.content)
            print(f"[{name}] done") 
        except Exception as e:
            #post the error anyway so the other workers don't wait forever
            await board.post(name, f"ERROR: {e}")


def find_python_code(text: str, hints: List[str]):
   
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", text, re.DOTALL)#find every fenced block and grabs the language tag + code works and multiline works because of re.DOTALL
    python_blocks = [body for lang, body in blocks if lang.lower() in ("python", "py", "python3", "")] #only keep blocks tagged as python or no tag


    if not python_blocks:
        if not blocks and ("def " in text or "class " in text): #if no fenced blocks but it looks like code, return the whole text
            return text.strip()
        return None


    for body in python_blocks:#prefer the first block that has a hint in it
        if any(hint in body for hint in hints): #if any of the hints are in the code, return it
            return body.strip()
    return python_blocks[0].strip() #if no hints matched, return the first python block


#since models may try to import thingfs from files that dont exist, comment out imports whos module cant be found. this is still safe because the real code gets pasted below the tests anyways
def strip_missing_imports(code: str) -> str:
    fixed = []
    for line in code.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            module_name = line.split()[1].split(".")[0].rstrip(",") #get the first part of the module name (rstrip handles "import os, sys")
            #relative imports like "from . import x" give an empty name, so those count as missing too
            if module_name == "" or importlib.util.find_spec(module_name) is None: #if the module cant be found, comment it out
                fixed.append(f"# {line}  #commented out because module not found")
                continue
        fixed.append(line)
    return "\n".join(fixed)

async def verity_worker(board: Blackboard): #uhh lowkey typed in verify wrong as verity but it still works so STICKING WITH IT 
    await board.wait_until(lambda: board.status["code"]=="completed" and board.status["tests"]=="completed")

    
    if await board.claim("verity"):
        print("[verity] running tests...")
        code = find_python_code(board.sections["code"], hints=["def ", "class "])
        tests = find_python_code(board.sections["tests"], hints=["def test_", "class Test"])
        if not code or not tests:
            await board.post("verity", "FAIL\nERROR: Could not find code or tests to run.")
            print("[verity] done")
            return

        #only strip the tests, if the real code imports something missing we WANT that to fail
        tests = strip_missing_imports(tests)

        #tests go first and the real code goes last. if the tests pasted their own copy of the function the real one overwrites it, and test functions look it up when they run so they use the real one
        combined = tests + "\n\n" + code + "\n"

        #use pytest if its installed, otherwise fall back to unittest (the tests prompt asks for unittest anyways)
        if importlib.util.find_spec("pytest") is not None:
            cmd = [sys.executable, "-m", "pytest", "-q", "test_generated.py"]
        else:
            cmd = [sys.executable, "-m", "unittest", "test_generated", "-v"]

        timed_out = False
        #create a temporary folder to hold the file, it deletes itself when the with block ends
        with tempfile.TemporaryDirectory() as tmpdirname:
            Path(tmpdirname, "test_generated.py").write_text(combined, encoding="utf-8")

            #run the tests in a separate python process so broken model code cant crash our script
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL, #no keyboard input so input() cant hang forever
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, #merge errors into the same output
                cwd=tmpdirname
            )
            try:
                #give up after 15 seconds so an infinite loop cant freeze everything
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                await proc.wait()
                stdout = b""

        text = stdout.decode(errors="replace")
        #decide the verdict from what actually happened, not from a models opinion
        if timed_out:
            verdict = "TIMEOUT"
            text = "tests ran longer than 15s (infinite loop?)"
        elif proc.returncode == 5 or "Ran 0 tests" in text: #pytest uses exit code 5 for "no tests collected"
            verdict = "NO TESTS" #nothing ran so this is not a pass
        elif proc.returncode == 0:
            verdict = "PASS" #exit code 0 means every test passed
        else:
            verdict = "FAIL"
        await board.post("verity", f"{verdict}\n{text}")
        print("[verity] done")


#entry point for testing
async def main():
    board = Blackboard(user_request="Write a Python function parse_duration(s) that turns strings like '2h30m' or '45s' into total seconds.")

    #create a list of model workers with their configurations
    models = [
        model_worker(board, "code"),
        model_worker(board, "tests"),
        model_worker(board, "docs"),
        model_worker(board, "security"),
        verity_worker(board)
    ]

    await asyncio.gather(*models) #run all models concurrently

    #print each section on its own so it's readable
    for name, content in board.sections.items():
        print(f"\n===== {name.upper()} =====\n{content}")

    print("\nEvents:")
    print(board.log)
    print("Done:", board.is_done())

if __name__ == "__main__":
    asyncio.run(main())
