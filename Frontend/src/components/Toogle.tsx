import React from "react";

interface ToggleProps {
    checked?: boolean;
    onChange?: (checked: boolean) => void;
    className?: string;
    label?: string;
}

const Toggle: React.FC<ToggleProps> = ({
    checked = false,
    onChange,
    className = "",
    label,
}) => {
    return (
        <label
            className={`inline-flex items-center cursor-pointer ${className}`}
        >
            {label && (
                <span className="mx-3 text-lg sm:text-xl tracking-wide font-medium text-text text-wrap">
                    {label}
                </span>
            )}
            <input
                type="checkbox"
                checked={checked}
                onChange={(e) => onChange?.(e.target.checked)}
                className="sr-only peer"
            />
            <div
                className="relative w-13 h-7 bg-gray-300 rounded-full peer
					peer-checked:bg-gray-900
					after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white 
					after:border after:border-gray-300 after:rounded-full after:h-6 after:w-6 
					after:transition-all peer-checked:after:translate-x-full 
					rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white"
            ></div>
        </label>
    );
};

export default Toggle;
