"""Normalize CPython opcode-event prefixes without changing edge identities."""
import dis


def prefix_aliases(code):
    instructions=list(dis.get_instructions(code));aliases={}
    pending=[]
    for instruction in instructions:
        if instruction.opname=='EXTENDED_ARG':pending.append(instruction.offset)
        else:
            for prefix in pending:aliases[prefix]=instruction.offset
            pending=[]
    if pending:raise ValueError('dangling_extended_arg')
    return aliases
