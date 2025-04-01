import React from "react";

interface DropdownProps {
    label: string;
    options: { value: string; label: string }[];
    selectedValue: string;
    onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
}

const Dropdown: React.FC<DropdownProps> = ({
    label,
    options,
    selectedValue,
    onChange,
}) => {
    return (
        <div className="py-2 flex justify-start items-center text-lg tracking-wide">
            <label className="bg-transparent px-2">{label}</label>
            <div className="bg-dark text-text rounded-2xl px-4 py-2">
                <select
                    className="bg-dark text-text"
                    value={selectedValue}
                    onChange={onChange}
                >
                    {options.map((option) => (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    ))}
                </select>
            </div>
        </div>
    );
};

export default Dropdown;
